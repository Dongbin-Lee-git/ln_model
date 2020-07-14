import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def forward_propagation(fir: numpy.array, sec: numpy.array):
    return sigmoid(numpy.dot(fir, sec))


def backward_propagation(back, w):
    return numpy.dot(w.T/sum(w.T), back)

def weight_update(error, w, o):
    return -(error * ((sigmoid(numpy.dot(w, o))) * (1 - sigmoid(numpy.dot(w, o)))) * o.T)

class neuralNetwork:
    def __init__(self, inum, hnum, onum, w_ih, w_ho):
        self.wih = w_ih
        self.who = w_ho
        print('\n{} * {} * {} 신경망 \n\nW_input_hidden : \n{}\n \nW_hidden_output : \n{}'.format(
                inum, hnum, onum, self.wih, self.who))
        pass

    def train(self, inputs_list, targets_list, ch, breakpoint):
        # 입력 리스트를 2차원의 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T
        if targets_list is not None:
            targets = numpy.array(targets_list, ndmin=2).T

        # 은닉 계층으로 들어오는 신호를 계산
        hidden_outputs = forward_propagation(self.wih, inputs)
        print('\nhidden_outputs : \n', hidden_outputs)
        # 최종 출력 계층으로 들어오는 신호를 계산
        final_outputs = forward_propagation(self.who, hidden_outputs)
        print('\nfinal_outputs : \n', final_outputs)

        if breakpoint in [1, 4]:
            print("\n-----↑{}번------\n".format(breakpoint))
            return


        print('\n 학습률 : {}\n'.format(ch))

        # 오차는 (실제 값 = 계산 값)
        final_errors = targets - final_outputs

        print('\nfinal_errors : \n', final_errors)

        # 은닉 계층의 오차는 weight 에 의해 나뉜 출력 계층의 오차들을 재조합해 계산
        hidden_errors = backward_propagation(final_errors, self.who)
        print("\nhidden_errors : \n", hidden_errors)

        # 은닉 계층과 출력 계층 간의 weight 업데이트
        self.who -= weight_update(final_errors, self.who, hidden_outputs) * ch
        print('\nW_hiddon_output_updated : \n', self.who)

        # 입력 계층과 은닉 계층 간의 weight 업데이트
        self.wih -= weight_update(hidden_errors, self.who, inputs) * ch
        print('\nW_input_hidden_updated : \n', self.wih)

        if breakpoint is 5:
            print("\n-----↑{}번------\n".format(breakpoint))
            return

#학습률 = ch(임의)
ch = 0.1
input_array = np.array([1.0, 0.5])

I333 = numpy.array([0.9, 0.1, 0.8])
I232 = numpy.array([0.9, 0.8])
W_input_hidden333 = numpy.array([[0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]])
W_input_hidden232 = numpy.array([[0.9, 0.4], [0.2, 0.2], [0.1, 0.6]])

W_hidden_output333 = numpy.array([[0.3, 0.7, 0.5], [0.6, 0.5, 0.-2], [0.8, 0.1, 0.9]])
W_hidden_output232 = numpy.array([[0.3, 0.7, 0.5],  [0.8, 0.1, 0.9]])

targets333 = numpy.array([0.01, 0.01, 0.99])
# targets232 = numpy.array([0.01, 0.99])

n333 = neuralNetwork(3, 3, 3, W_input_hidden333, W_hidden_output333)
n232 = neuralNetwork(2, 3, 2, W_input_hidden232, W_hidden_output232)

print("\n--- 333, 232 newralNetwork setting complete ---\n")


""" 1번 """
n333.train(I333, None, None, 1)


""" 2번 - backward propagation """
e_output = numpy.array([0.8, 0.5])
w_hidden_to_output = numpy.array([[2.0, 3.0], [1.0, 4.0]])
w_input_to_hidden = numpy.array([[3.0, 2.0], [1.0, 7.0]])
a = backward_propagation(e_output, w_hidden_to_output)
print(a)
print(backward_propagation(a, w_input_to_hidden))
print("\n-----↑2번------\n")


""" 3번 """
o = numpy.array([0.4, 0.5])
print(w_hidden_to_output - weight_update(e_output, w_hidden_to_output, o) * ch)
print("\n-----↑3번------")


""" 4번 """
n232.train(I232, None, None, 4)


""" 5번 """
n333.train(I333, targets333, 0.1, 5)