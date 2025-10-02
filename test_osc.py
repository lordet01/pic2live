"""OSC 통신 테스트 스크립트

TouchDesigner가 실행 중일 때 이 스크립트를 실행하여
OSC 메시지가 정상적으로 전송되는지 테스트합니다.
"""

from pythonosc import udp_client
import time
import math

def test_osc_connection():
    print("="*50)
    print("OSC 통신 테스트 시작")
    print("="*50)
    print("TouchDesigner가 실행 중이고 OSC In CHOP이 설정되어 있는지 확인하세요.")
    print("포트: 9000")
    print()
    
    # OSC 클라이언트 생성
    client = udp_client.SimpleUDPClient("127.0.0.1", 9000)
    
    print("테스트 메시지 전송 중... (10초간)")
    print("-"*50)
    
    try:
        for i in range(100):
            # 사인파 패턴으로 테스트 데이터 생성
            t = i / 10.0
            x = (math.sin(t) + 1.0) / 2.0  # 0~1 범위
            y = (math.cos(t) + 1.0) / 2.0  # 0~1 범위
            magnitude = abs(math.sin(t * 2))
            
            # OSC 메시지 전송
            client.send_message("/hand/present", 1)
            client.send_message("/hand/position/x", x)
            client.send_message("/hand/position/y", y)
            client.send_message("/effect/magnitude", magnitude)
            client.send_message("/effect/mode", (i // 10) % 9 + 1)
            
            # 진행 상황 출력
            if i % 10 == 0:
                print(f"[{i:3d}/100] x={x:.3f}, y={y:.3f}, mag={magnitude:.3f}")
            
            time.sleep(0.1)
        
        print("-"*50)
        print("✅ 테스트 완료!")
        print()
        print("TouchDesigner에서 OSC In CHOP의 값들이 변화했다면 성공입니다.")
        
    except KeyboardInterrupt:
        print("\n테스트 중단됨")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    test_osc_connection()


