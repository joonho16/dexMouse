"""Open-position calibration tests; no ROS installation or serial device needed."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


def load_driver():
    rclpy = types.ModuleType('rclpy')
    node = types.ModuleType('rclpy.node')
    node.Node = object
    sensor_msgs = types.ModuleType('sensor_msgs')
    msg = types.ModuleType('sensor_msgs.msg')
    msg.JointState = Mock
    modules = {'rclpy': rclpy, 'rclpy.node': node, 'sensor_msgs': sensor_msgs,
               'sensor_msgs.msg': msg, 'serial': types.ModuleType('serial')}
    source = Path(__file__).resolve().parents[1] / 'hand_serial_driver.py'
    spec = importlib.util.spec_from_file_location('driver_under_test', source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


driver_module = load_driver()
HandSerialDriver = driver_module.HandSerialDriver
DEFAULT_RANGES = {0: [0, 1000], 1: [0, 1270], 2: [0, 900],
                  3: [0, 900], 4: [0, 900], 5: [0, 900]}
OPEN_SAMPLE = (500, 45, 850, 890, 800, 920)


def packet_for(values, valid=True):
    payload = struct.pack('<6H', *values)
    checksum = sum(payload) & 0xFF
    if not valid:
        checksum ^= 1
    return b'\xff\xfe' + payload + bytes([checksum, 0x0A])


class OpenCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.driver = self.make_driver()

    def make_driver(self):
        driver = HandSerialDriver.__new__(HandSerialDriver)
        driver.SENSOR_RANGES = copy.deepcopy(DEFAULT_RANGES)
        driver.REVERSE_LIST = [2, 3, 4, 5]
        driver.calibration_path = Path(self.temp.name) / 'hand_open_calibration.json'
        driver.calibration_requested_at = None
        driver.keyboard_timer = Mock()
        driver.get_logger = Mock(return_value=Mock())
        driver.publish_goal_joint_state = Mock()
        driver.send_feedback_to_stm32 = Mock()
        driver.ser = Mock(in_waiting=0)
        driver.buffer = b''
        return driver

    def test_open_maps_to_zero_without_changing_closed_or_aa(self):
        self.driver.calibrate_open_hand(OPEN_SAMPLE)
        self.assertEqual(self.driver.SENSOR_RANGES[0], DEFAULT_RANGES[0])
        self.assertEqual(self.driver.normalize_aa_value(500, 0), 0.0)
        for idx in range(1, 6):
            with self.subTest(sensor=idx):
                self.assertEqual(self.driver.normalize_value(OPEN_SAMPLE[idx], idx), 0.0)
                self.assertEqual(self.driver.denormalize_value(0.0, idx), OPEN_SAMPLE[idx])
                closed = 1270 if idx == 1 else 0
                self.assertEqual(self.driver.normalize_value(closed, idx), 1.0)
                self.assertEqual(self.driver.denormalize_value(1.0, idx), closed)
                middle = (closed + OPEN_SAMPLE[idx]) / 2
                self.assertAlmostEqual(self.driver.normalize_value(middle, idx), 0.5)

    def test_saved_values_reload_and_repeat_calibration_does_not_accumulate(self):
        self.driver.calibrate_open_hand(OPEN_SAMPLE)
        reopened = self.make_driver()
        reopened.load_open_calibration()
        self.assertEqual(reopened.SENSOR_RANGES, self.driver.SENSOR_RANGES)
        reopened.calibrate_open_hand(OPEN_SAMPLE)
        self.assertEqual(reopened.SENSOR_RANGES, self.driver.SENSOR_RANGES)
        reopened.calibrate_open_hand((500, 30, 870, 880, 840, 900))
        self.assertEqual(reopened.SENSOR_RANGES[4], [0, 840])
        self.assertEqual(reopened.SENSOR_RANGES[1], [30, 1270])

    def test_key_waits_for_valid_packet_and_applies_only_once(self):
        with patch.object(driver_module.sys, 'stdin', io.StringIO('c\n')), \
                patch.object(driver_module.select, 'select', return_value=([sys.stdin], [], [])):
            self.driver.read_keyboard_callback()
        self.assertIsNotNone(self.driver.calibration_requested_at)
        self.assertFalse(self.driver.calibration_path.exists())
        self.driver.parse_packet(packet_for(OPEN_SAMPLE, valid=False))
        self.assertIsNotNone(self.driver.calibration_requested_at)
        self.assertFalse(self.driver.calibration_path.exists())
        self.driver.parse_packet(packet_for(OPEN_SAMPLE))
        self.assertIsNone(self.driver.calibration_requested_at)
        self.assertEqual(self.driver.SENSOR_RANGES[4], [0, 800])
        saved = self.driver.calibration_path.read_text()
        self.driver.parse_packet(packet_for((500, 500, 400, 400, 400, 400)))
        self.assertEqual(self.driver.calibration_path.read_text(), saved)
        self.driver.ser.write.assert_not_called()  # calibration sends no special serial command

    def test_other_key_does_not_calibrate(self):
        with patch.object(driver_module.sys, 'stdin', io.StringIO('x\n')), \
                patch.object(driver_module.select, 'select', return_value=([sys.stdin], [], [])):
            self.driver.read_keyboard_callback()
        self.driver.parse_packet(packet_for(OPEN_SAMPLE))
        self.assertEqual(self.driver.SENSOR_RANGES, DEFAULT_RANGES)
        self.assertFalse(self.driver.calibration_path.exists())

    def test_missing_sensor_data_times_out_without_using_old_position(self):
        self.driver.calibration_requested_at = 1.0
        with patch.object(driver_module.time, 'monotonic', return_value=2.1):
            self.driver.read_serial_callback()
        self.driver.parse_packet(packet_for(OPEN_SAMPLE))
        self.assertIsNone(self.driver.calibration_requested_at)
        self.assertEqual(self.driver.SENSOR_RANGES, DEFAULT_RANGES)
        self.assertFalse(self.driver.calibration_path.exists())

    def test_invalid_endpoint_keeps_all_previous_values_and_file(self):
        self.driver.calibrate_open_hand(OPEN_SAMPLE)
        before_ranges = copy.deepcopy(self.driver.SENSOR_RANGES)
        before_file = self.driver.calibration_path.read_text()
        for values in ((500, 1270, 800, 800, 800, 800),
                       (500, 0, 800, 800, 0, 800),
                       (500, 0, 800, 800, 4096, 800)):
            self.driver.calibrate_open_hand(values)
            self.assertEqual(self.driver.SENSOR_RANGES, before_ranges)
            self.assertEqual(self.driver.calibration_path.read_text(), before_file)

    def test_save_failure_keeps_old_calibration_and_removes_temporary_file(self):
        self.driver.calibrate_open_hand(OPEN_SAMPLE)
        before_ranges = copy.deepcopy(self.driver.SENSOR_RANGES)
        before_file = self.driver.calibration_path.read_text()
        with patch.object(Path, 'replace', side_effect=OSError('disk error')):
            self.driver.calibrate_open_hand((500, 0, 850, 890, 820, 920))
        self.assertEqual(self.driver.SENSOR_RANGES, before_ranges)
        self.assertEqual(self.driver.calibration_path.read_text(), before_file)
        self.assertEqual(list(Path(self.temp.name).glob('*.tmp')), [])

    def test_missing_or_invalid_saved_calibration_keeps_defaults(self):
        self.driver.load_open_calibration()
        self.assertEqual(self.driver.SENSOR_RANGES, DEFAULT_RANGES)
        positions = {str(idx): OPEN_SAMPLE[idx] for idx in range(1, 6)}
        for invalid in ('{', '[]', '{}', json.dumps({**positions, '4': True}),
                        json.dumps({**positions, '4': 0})):
            self.driver.calibration_path.write_text(invalid)
            self.driver.load_open_calibration()
            self.assertEqual(self.driver.SENSOR_RANGES, DEFAULT_RANGES)

    def test_feedback_uses_the_same_calibrated_open_endpoints(self):
        self.driver.calibrate_open_hand(OPEN_SAMPLE)
        self.driver.last_joint_msg = types.SimpleNamespace(
            name=['finger1_AA', 'finger1_FE', 'finger2_FE', 'finger3_FE', 'finger4_FE'],
            position=[0.0] * 5)
        HandSerialDriver.send_feedback_to_stm32(self.driver)
        packet = self.driver.ser.write.call_args.args[0]
        self.assertEqual(len(packet), 14)
        self.assertEqual(struct.unpack('<5H', packet[2:-2]), (0, 45, 850, 890, 800))
        self.assertEqual(packet[-2], sum(packet[2:-2]) & 0xFF)


if __name__ == '__main__':
    unittest.main()
