import matplotlib.pyplot as plt

VAL_LOSSES = [4.7477986812591553, 4.6491537094116211, 4.6092797517776489, 4.5777109861373901, 4.55376136302948, 4.5350325107574463, 4.5151046514511108, 4.5011823177337646, 4.4931303262710571, 4.482452392578125]
TRAIN_LOSSES = [5.1878022087944879, 4.8902433051003351, 4.7637442350387573, 4.6580495172076759, 4.6151861084832086, 4.5983320474624634, 4.5478902790281506, 4.5215979682074652, 4.5195024278428821, 4.4851922459072533]

def plot_losses():
	plt.plot(range(len(VAL_LOSSES)), VAL_LOSSES, label='Validation Loss')
	plt.plot(range(len(TRAIN_LOSSES)), TRAIN_LOSSES, label='Training Loss')
	plt.title('Loss Per Epoch')
	plt.xlabel('Epoch')
	plt.ylabel('Regularized loss')
	plt.legend()
	plt.show()
	#plt.savefig('loss_per_epoch.png', bbox_inches='tight')

if __name__ == "__main__":
	plot_losses()