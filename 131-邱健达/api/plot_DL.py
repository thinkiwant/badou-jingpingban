import matplotlib.pyplot as plt

def plotACC_testAndLOSS_train(accAndce):
    plt.plot(range(len(accAndce)), [ce[0] for ce in accAndce], label="ACC")
    plt.plot(range(len(accAndce)), [ce[1] for ce in accAndce], label="CE")
    plt.legend()
    plt.show()
    return