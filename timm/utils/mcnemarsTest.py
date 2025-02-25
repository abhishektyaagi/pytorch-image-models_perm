import torch
import torch.nn.functional as F

# Suppose you have something like:
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# modelA, modelB are already loaded on GPU, in eval mode, etc.

# Initialize counts
n00 = 0  # both wrong
n11 = 0  # both right
n10 = 0  # A right, B wrong
n01 = 0  # B right, A wrong

for images, labels in val_loader:
    images = images.cuda()
    labels = labels.cuda()

    # Get top-1 predictions for both models
    with torch.no_grad():
        logitsA = modelA(images)
        predsA = torch.argmax(logitsA, dim=1)
        logitsB = modelB(images)
        predsB = torch.argmax(logitsB, dim=1)

    correctA = (predsA == labels)
    correctB = (predsB == labels)

    # Convert to CPU Tensors of bool
    correctA = correctA.cpu()
    correctB = correctB.cpu()

    # Count outcomes
    for ca, cb in zip(correctA, correctB):
        if ca and cb:
            n11 += 1  # both correct
        elif not ca and not cb:
            n00 += 1  # both wrong
        elif ca and not cb:
            n10 += 1  # A right, B wrong
        else:
            n01 += 1  # B right, A wrong

from statsmodels.stats.contingency_tables import mcnemar

# Build 2x2 matrix
# By convention, the table is often arranged as:
# [[n_00, n_01],
#  [n_10, n_11]]
table = [[n00, n01],
         [n10, n11]]

# Use exact=False for the chi-square approximation,
# or exact=True for the binomial distribution test, which might be slow for large n
result = mcnemar(table, exact=False, correction=True)
print("statistic = ", result.statistic)
print("p-value   = ", result.pvalue)

if result.pvalue < 0.05:
    print("Significant difference between Model A & B (alpha=0.05).")
else:
    print("No significant difference between Model A & B.")
