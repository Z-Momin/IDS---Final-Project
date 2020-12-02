import matplotlib.pyplot as plt

mean = [-15.986544605854483, 
878.9312499999795, 
819.5999264291208, 
464.2065939656529,
866.6279189827603,
633.3532169693632]



std = [25.08045272364044,
58.310180487106955,
161.3520185864465,
225.93668155346353,
66.85069799243911, 
162.18001773798852]

labels = ["baseline", "human", "model A", "model B", "model C", "model D"]
x_pos = [i for i, _ in enumerate(labels)]

fig, ax = plt.subplots()
ax.bar(x_pos, mean, yerr=std, align='center')
ax.set_ylabel('score')
ax.set_xticklabels(labels)
ax.set_xticks(x_pos)
ax.set_title("Evaluation Performance")
ax.yaxis.grid(True)

plt.tight_layout()
plt.savefig('images/evaluation.png')
plt.show()