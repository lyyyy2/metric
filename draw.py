import matplotlib.pyplot as plt

model = ['I', 'I+T', 'I+C']
score = [[0.8046, 0.6859, 0.7453], [0.7879, 0.7590, 0.7735], [0.4854, 0.7363, 0.6109]]
# color_map = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}
c = ['ref', 'green', 'blue']
plt.figure(figsize=(10, 15), dpi=100)
plt.scatter(model, score, c=c*3)
plt.xlabel("Scores", fontdict={'size': 16})
plt.ylabel("Model", fontdict={'size': 16})
# plt.title("Metaphor evaluation results of ", fontdict={'size': 20})
# plt.legend(loc='best')
plt.show()
