import matplotlib.pyplot as plt

mean = [-15.986544605854483, 
878.9312499999795,
-93.00467776493731,
660.480939516178,
819.5999264291208, 
464.2065939656529,
866.6279189827603,
633.3532169693632]



std = [25.08045272364044,
58.310180487106955,
0.5818251797978774,
255.41402587315116,
161.3520185864465,
225.93668155346353,
66.85069799243911, 
162.18001773798852]

labels = ["Baseline", "Human", "LR", "RF", "DL-Model A", "DL-Model B", "DL-Model C", "DL-Model D"]
x_pos = [i for i, _ in enumerate(labels)]

fig, ax = plt.subplots()
ax.bar(x_pos, mean, yerr=std, align='center')
ax.set_ylabel('Score')
ax.set_xticklabels(labels)
ax.set_xticks(x_pos)
ax.set_title("Evaluation Performance")
ax.yaxis.grid(True)

plt.tight_layout()
plt.savefig('images/evaluation.png')
plt.show()