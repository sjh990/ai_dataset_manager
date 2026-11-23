"""시설물 라벨링 툴 패키지.

윈도우에서 torch 가 시작할 때 띄우는 경고 2가지 숨김 (툴 동작과 무관, 윈도우라서 나오는 안내):
  - triton not found (리눅스 전용 GPU 부품, 윈도우 torch 에는 원래 없음)
  - Redirects are currently not supported in Windows or MacOs
torch 를 import 하기 전에 실행되어야 해서 여기(labeler 를 처음 import 할 때)에 둠.
"""
import logging

for _name in ("torch.utils.flop_counter", "torch.distributed.elastic.multiprocessing.redirects"):
    logging.getLogger(_name).setLevel(logging.ERROR)
