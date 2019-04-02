import argparse
import numpy as np
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--train',
                    help="path to gop train file")
parser.add_argument('--test',
                    help="path to gop test file")
args = parser.parse_args()

X = []
y = []

with open(args.train) as file_gop:
    for l in file_gop:
        l = l.strip()
        p = l.find("[")
        filename = l[:p].strip()
        q = l.find("]")
        feats = [float(s) for s in l[p+1:q].strip().split(" ")]

        npfeats = np.array(feats)

        if filename.startswith("KR"):
            label = 1
        else:
            label = 0

        X.append([np.median(npfeats), np.max(npfeats), np.mean(npfeats), np.average(npfeats)])
        y.append(label)

reg = LinearRegression().fit(np.array(X), np.array(y))

print "coefficient of determination R^2 of the prediction", reg.score(X, y)

with open(args.test) as file_gop:
    for l in file_gop:
        X = []
        l = l.strip()
        p = l.find("[")
        filename = l[:p].strip()
        q = l.find("]")
        feats = [float(s) for s in l[p+1:q].strip().split(" ")]

        npfeats = np.array(feats)

        if filename.startswith("KR"):
            label = 1
        else:
            label = 0

        X.append([np.median(npfeats), np.max(npfeats), np.mean(npfeats), np.average(npfeats)])
        print filename, label, reg.predict(X)
