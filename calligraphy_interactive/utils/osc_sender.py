from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
from pythonosc import udp_client


class OSCSender:
    """TouchDesigner와 OSC 프로토콜로 통신하는 송신 클래스"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 9000, enabled: bool = True) -> None:
        self.host = host
        self.port = port
        self.enabled = enabled
        self.client: Optional[udp_client.SimpleUDPClient] = None
        
        if self.enabled:
            try:
                self.client = udp_client.SimpleUDPClient(self.host, self.port)
                print(f"OSC Sender initialized: {self.host}:{self.port}")
            except Exception as e:
                print(f"OSC Sender failed to initialize: {e}")
                self.enabled = False
    
    def send_gesture_state(self, state, ctrl_left: Dict[str, Any], ctrl_right: Dict[str, Any]) -> None:
        """양손 제스처 상태를 TouchDesigner로 전송"""
        if not self.enabled or self.client is None:
            return
        
        try:
            # 왼손 데이터
            self.client.send_message("/hand/left/present", 1 if state.left_hand.present else 0)
            if state.left_hand.present:
                self.client.send_message("/hand/left/position/x", float(state.left_hand.position_norm[0]))
                self.client.send_message("/hand/left/position/y", float(state.left_hand.position_norm[1]))
                self.client.send_message("/hand/left/velocity/x", float(state.left_hand.velocity_norm[0]))
                self.client.send_message("/hand/left/velocity/y", float(state.left_hand.velocity_norm[1]))
                
                # 왼손 이펙트 컨트롤 데이터
                dx, dy = ctrl_left['direction']
                self.client.send_message("/effect/left/direction/x", float(dx))
                self.client.send_message("/effect/left/direction/y", float(dy))
                self.client.send_message("/effect/left/magnitude", float(ctrl_left['magnitude']))
                
                # 왼손 엄지-검지 거리와 각도
                self.client.send_message("/hand/left/thumb_index/distance_cm", float(state.left_hand.thumb_index_distance_cm))
                self.client.send_message("/hand/left/thumb_index/distance_norm", float(state.left_hand.thumb_index_distance_norm))
                self.client.send_message("/hand/left/thumb_index/angle_rad", float(state.left_hand.thumb_index_angle_rad))
                
                # 왼손 랜드마크 데이터 (index finger tip: 8번)
                if state.left_hand.landmarks_norm and len(state.left_hand.landmarks_norm) > 8:
                    tip_x, tip_y = state.left_hand.landmarks_norm[8]
                    self.client.send_message("/hand/left/fingertip/x", float(tip_x))
                    self.client.send_message("/hand/left/fingertip/y", float(tip_y))
            
            # 오른손 데이터
            self.client.send_message("/hand/right/present", 1 if state.right_hand.present else 0)
            if state.right_hand.present:
                self.client.send_message("/hand/right/position/x", float(state.right_hand.position_norm[0]))
                self.client.send_message("/hand/right/position/y", float(state.right_hand.position_norm[1]))
                self.client.send_message("/hand/right/velocity/x", float(state.right_hand.velocity_norm[0]))
                self.client.send_message("/hand/right/velocity/y", float(state.right_hand.velocity_norm[1]))
                
                # 오른손 이펙트 컨트롤 데이터
                dx, dy = ctrl_right['direction']
                self.client.send_message("/effect/right/direction/x", float(dx))
                self.client.send_message("/effect/right/direction/y", float(dy))
                self.client.send_message("/effect/right/magnitude", float(ctrl_right['magnitude']))
                
                # 오른손 엄지-검지 거리와 각도
                self.client.send_message("/hand/right/thumb_index/distance_cm", float(state.right_hand.thumb_index_distance_cm))
                self.client.send_message("/hand/right/thumb_index/distance_norm", float(state.right_hand.thumb_index_distance_norm))
                self.client.send_message("/hand/right/thumb_index/angle_rad", float(state.right_hand.thumb_index_angle_rad))
                
                # 오른손 랜드마크 데이터 (index finger tip: 8번)
                if state.right_hand.landmarks_norm and len(state.right_hand.landmarks_norm) > 8:
                    tip_x, tip_y = state.right_hand.landmarks_norm[8]
                    self.client.send_message("/hand/right/fingertip/x", float(tip_x))
                    self.client.send_message("/hand/right/fingertip/y", float(tip_y))
            
            # 하위 호환성을 위한 통합 데이터 (주 손 = 오른손 우선)
            self.client.send_message("/hand/present", 1 if state.any_hand_present else 0)
            self.client.send_message("/hand/position/x", float(state.position_norm[0]))
            self.client.send_message("/hand/position/y", float(state.position_norm[1]))
            self.client.send_message("/hand/velocity/x", float(state.velocity_norm[0]))
            self.client.send_message("/hand/velocity/y", float(state.velocity_norm[1]))
            
            # 주 손의 이펙트 데이터
            main_ctrl = ctrl_right if state.right_hand.present else ctrl_left
            dx, dy = main_ctrl['direction']
            self.client.send_message("/effect/direction/x", float(dx))
            self.client.send_message("/effect/direction/y", float(dy))
            self.client.send_message("/effect/magnitude", float(main_ctrl['magnitude']))
        
        except Exception as e:
            # 통신 실패 시 조용히 무시 (성능 저하 방지)
            pass
    
    def send_custom(self, address: str, value: Any) -> None:
        """커스텀 OSC 메시지 전송"""
        if not self.enabled or self.client is None:
            return
        
        try:
            self.client.send_message(address, value)
        except Exception:
            pass
    
    def close(self) -> None:
        """리소스 정리"""
        self.client = None
        self.enabled = False




