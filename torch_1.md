```python
from sklearn import load_digits
dataset = load_digits(n_class = 10)
```
- float로 이루어진 데이터라면 torch.FloatTensor()로 묶고
- int면 torch.LongTensor()

- 전체 데이터 중에서 하나의 행이 무슨 클래스에 속하는지 맞추는 task
- nn.Linear()는 선형사상  #TODO 선형사상 복습 #왜 선형사상을 해야 하는 것일까?
- "nn.tanh()는 선형사상 뒤에 일단 먹여라: 이유는 지금 말하지 못하는데 걍 그렇게 하는게 좋아"
  - 이건 차원을 줄이거나 하지 않으므로
- ```nn.Linear(in_size, h_size)```는 차원을 h_size만큼 줄이는 작업
- softmax를 ```nn.CrossEntropyLoss(reduction='sum')```로 불러왔네
  - reduction?
  - CrossEntropyLoss()안에는 Softmax()가 포함되어있기 때문에 와다상은 한꺼번에 돌려버렸는데 따로 따로 돌릴 때랑 동시에 ```softmax = nn.CrossEntropyLoss
  (reduction = 'sum')```처럼 적는 것은 모델 적으로 뭔가 다른지 알아봐야
  - softmax는 한 행의 합이 1.0이 되도록 하는 함수 (확률로 바꾸는): 10개의 클래스를 맞추는 task에서 각 클래스에 대한 확률이 각 값이 됨: 그러므로 가장 높은 확률을 갖는 클래스가 해당 
  행이 속하는 클래스로 추측

- .zero_grad()저번에 한 계산을 초기화
- ```torch.no_grad()```는 이 이후부터는 gradient계산을 하지 않는다는 얘기
- copy.deepcopy(model)는 pickle처럼 code와 같이 binary로 저장하지만 파일로는 전혀
- no_grad()는 gradient를 하지 않는데 torch.eval()은 nn.dropout()같은 함수를 훈련 때만 혹은 테스트 때만 자동적으로 분별해서 사용해주기 때문에 개인적으로는 2 개 동시에 
적어두는게 나을 듯

- 와다상은 하나의 클래스 안에 여러 함수를 만들어서 훈련용이랑 테스트용을 같이 넣어두니까 model.train()했다가 model.eval()했다가 하고 있음

- ```nn.Sequential()```은 자동으로 레이어 수를 정할 수 있다고?
  - 아 for문으로 range를 돌렸네
  - 원라이너로 가능
  - 안에 있는 레이어를 따로 따로 부르는 것은 불가능
  
- ```nn.Modulelist()```는 인덱스로 레이어를 하나 하나 불러올 수 있음: 나중을 생각하면 이게 좋음
  - 그러니까 이건 ```for i in range(len(self.layers))```로 하면 됨
  - 오자키: 하나로 묶지 않으면 파라메터가 이랬다 저랬다 하니 묶는게 좋은데 nn.Modulelist()는 for문으로 돌려야 하니까 기분 나쁨
  
- 호리구치: 학습할 때 require_grad를 False로 해 놓지 않으면 embedding layer까지 학습해버리니까?
  - 와다상: pre_trained를 사용할 때는 또 다르지만
  
- attention할 때에는 lstm의 output(hs)를 사용
  - bi-lstm의 hs에는 모든 스텝 포함되어있고 h_n(hidden_state), c_n(cell_state)은 최종 레이어 정보가 들어가 있음
  - 레이어가 2개인 bi-lstm의 h_n에는 0번 레이어 순방향 최종값, 0번 레이어 역방향 최종값, 1번 레이어 순방향 최종값, 1번 레이어 역방향 최종값이 들어있음
  - 예를 들어, 1번 레이어 순방향 최종값은 ```h_n[1, 0, :, :]```
  - ```hs[0][0]```라면 마지막 레이어의 순방향 처음과 역방향 마지막 값, ```hs[0][-1]```라면 마지막 레이어의 순방향 마지막값과 역방향 첫 값
  - 두 개가 제대로 일치하는지 확인하는 작업에서 사용
  
- lstm할 때에는 pack, pad만 하는데 attention할 때에는 masking하는 등의 작업이 필요
  - 가장 문장 길이가 긴 순서대로 batch를 만들지 않으면 계산할 때 よろしくない