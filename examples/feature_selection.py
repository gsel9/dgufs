from dgufs.dgufs import DGUFS

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler

    iris = load_iris(return_X_y=False)

    scaler = StandardScaler()

    X, y = iris.data, iris.target
    X_std = scaler.fit_transform(X)

    dgufs = DGUFS(num_features=2, num_clusters=3)
    dgufs.fit(X_std)

    print(dgufs.memberships)
    print(dgufs.support)

    # Selected features
    # X_sub = X[:, dgufs.support]
    # print(X_sub.shape)
