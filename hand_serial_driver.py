#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import serial
import struct
import time

# --- [Protocol] STM32 통신 패킷 설정 ---
# 수신 (STM32 -> PC)
RX_PACKET_SIZE = 16
RX_HEADER_1 = 0xFF
RX_HEADER_2 = 0xFE
TAIL_BYTE = 0x0A

# 송신 (PC -> STM32) : 로봇 상태 피드백용
TX_HEADER_1 = 0xFF
TX_HEADER_2 = 0xFD # 수신과 구분하기 위해 다른 헤더 사용

class HandSerialDriver(Node):
    def __init__(self):
        super().__init__('hand_serial_driver')
        
        # 1. 파라미터 설정
        self.declare_parameter('port', '/dev/ttyUSB1')
        self.declare_parameter('baudrate', 1000000)
        
        port_name = self.get_parameter('port').value
        baud_rate = self.get_parameter('baudrate').value
        
        # 2. 시리얼 포트 개방
        try:
            self.ser = serial.Serial(port_name, baud_rate, timeout=0.05)
            self.get_logger().info(f"Connected to {port_name} at {baud_rate}bps")
        except Exception as e:
            self.get_logger().error(f"Failed to open serial port: {e}")
            exit(1)
            
        # 3. Publisher: 마스터 핸드 데이터 -> 로봇 제어 명령
        self.publisher_ = self.create_publisher(JointState, 'goal_joint_states', 10)
        
        # 4. Subscriber: 로봇 실제 상태 -> 마스터 핸드 피드백 (추가됨)
        self.create_subscription(JointState, 'joint_states', self.feedback_callback, 10)
        
        # 5. 수신 타이머
        self.create_timer(0.001, self.read_serial_callback)
        
        self.buffer = b''
        self.last_joint_msg = None # [추가] 가장 최신의 로봇 상태 저장용

        # --- [설정] 센서 보정 값 ---
        # 0:ThumbAA, 1:ThumbFE, 2:Index, 3:Middle, 4:Ring, 5:Pinky
        self.SENSOR_RANGES = {
            0: [0, 1000], # 엄지 회전
            1: [0, 1270], # 엄지 굽힘
            2: [0, 900], # 검지
            3: [0, 900], # 중지
            4: [0, 900], # 약지
            5: [0, 900]  # 소지 (사용 안 함)
        }

        # 반대로 작동하는 센서 인덱스 리스트
        self.REVERSE_LIST = [2, 3, 4]

    def read_serial_callback(self):
        if self.ser.in_waiting > 0:
            self.buffer += self.ser.read(self.ser.in_waiting)
            
            while len(self.buffer) >= RX_PACKET_SIZE:
                if self.buffer[0] == RX_HEADER_1 and self.buffer[1] == RX_HEADER_2:
                    packet = self.buffer[:RX_PACKET_SIZE]
                    if packet[-1] == TAIL_BYTE:
                        self.parse_packet(packet)
                        self.buffer = self.buffer[RX_PACKET_SIZE:]
                    else:
                        self.buffer = self.buffer[1:]
                else:
                    self.buffer = self.buffer[1:]

    def parse_packet(self, packet):
        try:
            # Unpacking: Header(2x) + 5 Sensors(5h) + Encoder(h) + Checksum(B) + Tail(x)
            data = struct.unpack('<2x 5h h B x', packet)
            
            raw_joints = data[0:6] 
            checksum_recv = data[6]
            
            payload = packet[2:-2]
            checksum_calc = sum(payload) & 0xFF
            
            if checksum_recv == checksum_calc:
                # 1. ROS로 명령 발행
                self.publish_goal_joint_state(raw_joints)
                
                # 2. [추가] 즉시 현재 로봇 상태를 STM32로 응답(Response)
                self.send_feedback_to_stm32()

        except Exception as e:
            self.get_logger().error(f"Parsing Error: {e}")

    # --- [값 변환 로직] ---
    
    # Raw(0~4095) -> Norm(0.0~1.0)
    def normalize_value(self, raw_val, sensor_idx):
        min_v, max_v = self.SENSOR_RANGES.get(sensor_idx, [0, 4095])
        val = max(min_v, min(raw_val, max_v))
        if max_v - min_v == 0: return 0.0
        norm = (val - min_v) / (max_v - min_v)
        
        if sensor_idx in self.REVERSE_LIST:
            norm = 1.0 - norm
        return norm

    # Norm(0.0~1.0) -> Raw(0~4095) [피드백 송신용]
    def denormalize_value(self, norm_val, sensor_idx):
        min_v, max_v = self.SENSOR_RANGES.get(sensor_idx, [0, 4095])
        
        # 반전 처리 복구
        if sensor_idx in self.REVERSE_LIST:
            norm_val = 1.0 - norm_val
            
        # 범위 클램핑
        norm_val = max(0.0, min(1.0, norm_val))
        
        # 선형 변환
        raw = int(min_v + (max_v - min_v) * norm_val)
        return raw

    def normalize_aa_value(self, raw_val, sensor_idx):
        norm = self.normalize_value(raw_val, sensor_idx)
        return (norm * 2.0) - 1.0

    # Norm(-1.0~1.0) -> Raw(0~4095) [피드백 송신용]
    def denormalize_aa_value(self, norm_val, sensor_idx):
        # -1~1 -> 0~1 변환
        norm_0_1 = (norm_val + 1.0) / 2.0
        return self.denormalize_value(norm_0_1, sensor_idx)

    # --- [Subscriber] 로봇 상태 저장 ---
    def feedback_callback(self, msg):
        # 여기서는 저장만 하고, 전송은 시리얼 수신 시점에 함
        self.last_joint_msg = msg

    # --- [Serial Write] 저장된 상태 전송 ---
    def send_feedback_to_stm32(self):
        if self.last_joint_msg is None:
            return

        msg = self.last_joint_msg
        
        # 1. 메시지를 딕셔너리로 변환 (이름: 값)
        joint_map = {name: pos for name, pos in zip(msg.name, msg.position)}
        
        # 2. 필요한 데이터 추출 및 역변환 (Robot -> Master)
        try:
            # 매핑 규칙 (publish_goal_joint_state와 대칭)
            # [0] Thumb AA (finger1_AA)
            val_0 = self.denormalize_value(joint_map.get('finger1_AA', 0.0), 0)
            
            # [1] Thumb FE (finger1_FE)
            val_1 = self.denormalize_value(joint_map.get('finger1_FE', 0.0), 1)
            
            # [2] Index FE (finger2_FE)
            val_2 = self.denormalize_value(joint_map.get('finger2_FE', 0.0), 2)
            
            # [3] Middle FE (finger3_FE)
            val_3 = self.denormalize_value(joint_map.get('finger3_FE', 0.0), 3)
            
            # [4] Ring FE (finger4_FE)
            val_4 = self.denormalize_value(joint_map.get('finger4_FE', 0.0), 4)
            
            # [5] Pinky (Robot에는 없으므로 0 또는 더미값)
            val_5 = 0

            # 3. 패킷 생성
            # Header(2) + Data(5*Short) + Checksum(1) + Tail(1) = 14 Bytes (예시)
            data_payload = struct.pack('<5H', val_0, val_1, val_2, val_3, val_4)
            checksum = sum(data_payload) & 0xFF
            
            packet = struct.pack('<BB', TX_HEADER_1, TX_HEADER_2) + data_payload + struct.pack('<BB', checksum, TAIL_BYTE)
            
            # 4. 시리얼 전송
            self.ser.write(packet)
            
        except Exception as e:
            pass

    # --- [STM32 -> ROS] 마스터 핸드 명령 수신 ---
    def publish_goal_joint_state(self, raw_joints):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        msg.name = [
            'finger1_AA', 'finger1_FE', 
            'finger2_AA', 'finger2_FE', 
            'finger3_AA', 'finger3_FE', 
            'finger4_AA', 'finger4_FE'
        ]
        
        # 1. 엄지
        val_1_aa = self.normalize_aa_value(raw_joints[0], 0)
        val_1_fe = self.normalize_value(raw_joints[1], 1)
        
        # 2~4. 나머지 손가락
        val_2_fe = self.normalize_value(raw_joints[2], 2)
        val_3_fe = self.normalize_value(raw_joints[3], 3)
        val_4_fe = self.normalize_value(raw_joints[4], 4)
        
        msg.position = [
            float(val_1_aa), float(val_1_fe),
            0.0,             float(val_2_fe),
            0.0,             float(val_3_fe),
            0.0,             float(val_4_fe)
        ]
        
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = HandSerialDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()