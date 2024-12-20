def task_1(work_id):    

    import random
    from matplotlib import pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV    
    import pandas as pd    
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures

    random.seed(work_id)

    data1 = pd.read_csv('https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/microchip_tests.txt',
                   header=None, names = ('test1','test2','released'))

    X = data1.iloc[:,:2].values
    y = data1.iloc[:,2].values
    
    poly = PolynomialFeatures(degree=7)
    
    X_poly = poly.fit_transform(X)
    
    
    
    def plot_boundary(clf, X, y, grid_step=.01, poly_featurizer=None):
        x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
        y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                             np.arange(y_min, y_max, grid_step))
        Z = clf.predict(poly_featurizer.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
        
        
    random.seed(work_id)
    C = [1e-2, 10, 100000]
    random.shuffle(C)
    Q = random.choice(["overfited", "normal", "underfited"])
     
    logit = LogisticRegression(C=C[0], n_jobs=1, random_state=17, max_iter = 10000)
    logit.fit(X_poly, y)

    plot_boundary(logit, X, y, grid_step=.005, poly_featurizer=poly)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red')
    plt.title("Model 1")
    plt.show()

    logit = LogisticRegression(C=C[1], n_jobs=1, random_state=17, max_iter = 10000)
    logit.fit(X_poly, y)

    plot_boundary(logit, X, y, grid_step=.005, poly_featurizer=poly)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red')
    plt.title("Model 2")
    plt.show()

    logit = LogisticRegression(C=C[2], n_jobs=1, random_state=17, max_iter = 10000)
    logit.fit(X_poly, y)

    plot_boundary(logit, X, y, grid_step=.005, poly_featurizer=poly)

    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='green')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red')
    plt.title("Model 3")
    plt.show()
    
    print("There are 3 models (normal, overfited and unerfited). Chouse picture with "+Q+ " model. Provide your answer in form anwer_1 = number_of_choused_model" )
    
    
    
def task_2(work_id):
    import random

    questions = ["Determine if an email is spam based on existing database of emails marked \"Spam\" or \"Not Spam\"",
    "Predict the sales volumes of live Christmas trees in December based on their sales data for previous years",
    "Find out, based on data on paintings by famous artists of the 19th century, whether Vincent van Gogh is the author of a recently found painting",
    "Retrieve the age of a Twitter account owner using a database of 10,000 accounts with a known age",
    "Assess lung cancer risk based on patient data and patient records database",
    "Identify groups of students with similar abilities to distribute them into study groups",
    "Identify combinations of products that are often found in one check in a coffee shop",
    "Divide vacancies into groups based on ad texts",
    "Select groups of travel agency clients based on information about their order history",
    "Predict how long a trip will take, knowing the average taxi speed, route distance, traffic information and a database of previous trips",
    "Determine if a new text was written by Leonardo da Vinci, having a database of all his previous works",
    "Highlight Communities in the Facebook Friends Network"]
    random.seed(work_id)
    ids = [x for x in range(12)]
    random.shuffle(ids)
    print("Chouse numbers of tasks (there can be from 0 to 6) that represents supervised learning:")
    for i in range(6):
        print("Task "+str(i+1)+": " + questions[ids[i]])

#111110000110


def task_3(work_id):
    import random
    from sklearn import datasets
    import numpy as np
    digits = datasets.load_digits()
    random.seed(work_id)
    prc = random.choice([70, 75, 80, 85, 90])
    print("For given dataset evaluate the minimal number of components in PCA for preserving at least " + str(prc)+"% of variance.") 
    return(np.array(random.sample(list(digits.data), 1000)))



def task_4(work_id):
    import numpy as np
    import random
    from matplotlib import pyplot as plt
    from sklearn.cluster import KMeans



    random.seed(work_id)
    num = random.choice([4,5,6,7])
    cl_x = [0, 3, 2, 1,  3, -1, 1]
    cl_y = [-1, 2, 0, 1.5, -1, 3, -2]
    id_s = random.sample([0,1,2,3,4,5,6], num)
    
    X = np.zeros((420, 2))
    np.random.seed(seed=work_id)
  
    p = 420 // num
    
#    print(p, num)
    
    for ii, i in enumerate(id_s):
        
        X[ii*p:((ii+1)*(p)), 0] = np.random.normal(loc=cl_x[i], scale=.15, size=p)
        X[ii*p:((ii+1)*(p)), 1] = np.random.normal(loc=cl_y[i], scale=.15, size=p)


#    plt.figure(figsize=(5, 5))
#    plt.plot(X[:, 0], X[:, 1], 'bo');
#    plt.show()
#    print("gg")
    inertia = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=work_id).fit(X)
        inertia.append(np.sqrt(kmeans.inertia_))
        
    plt.plot(range(1, 10), inertia, marker='s');
    plt.xlabel('$k$')
    plt.ylabel('$J(C_k)$');
  
    print("For provided graph, chose optimal number of clusters")
    
    return(inertia)


def task_5(work_id):
    import random
    import numpy as np
    n_train = 150        
    n_test = 1000       
    noise = 0.1
    random.seed(work_id)
    
    x1, x2 = random.sample([0,1,2,3,4,5,6], 2)
    
    np.random.seed(seed=work_id) 
    
    
    # Generate data
    
    
#    def f(x):
#        x = x.ravel()
#        return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

    def generate(n_samples, noise, x1, x2):
        X = np.random.rand(n_samples) * 10 - 5
        X = np.sort(X).ravel()
        y = np.exp(-(X-x1) ** 2) + 1.5 * np.exp(-(X - x2) ** 2)\
            + np.random.normal(0.0, noise, n_samples)
        X = X.reshape((n_samples, 1))

        return X, y

    X_train, y_train = generate(n_samples=n_train, noise=noise, x1 = x1, x2 = x2)
    X_test, y_test = generate(n_samples=n_test, noise=noise, x1 = x1, x2 = x2)


    print("For provided test and train dataset fit DecisionTreeRegressor and RandomForestRegressor with n_estimators=10 and random_state = work_id on train")
    print("Calculate difference of MSE of this two models on test set")


    return(X_train, y_train, X_test, y_test)


def task_6(work_id):
    import random
    import pandas as pd
    
    random.seed(work_id)
    y = random.choice([10, 15, 20, 25, 30])
    males = random.choice(["males", "females"])
    print("For given dataset, find the number of "+ males +" strictly older than "+ str(y) +" years") 
    
    train = pd.read_csv("https://github.com/ipython-books/cookbook-2nd-data/blob/master/titanic_train.csv?raw=true", 
                       index_col='PassengerId') 
    return(train.sample(500, random_state = work_id))
    
    
def task_7(work_id):
    import random
    import pandas as pd
    train = pd.read_csv("https://raw.githubusercontent.com/AVSirotkin/data_for_classes/master/brca.csv")
    
    print("Remove first two columns.")
    print("Use column \"y\" as target variable.")
    print("Use RandomForestClassifier for prediction.")
    print("Based on cross validation chouse best cobination of parameters:")
    print("n_estimators: 10, 30, 70, 100")
    print("max_depth: 3, 5, 7, 10")
    print("min_samples_leaf: 1, 4, 7, 11, 15, 21")

    print("Use best combination, to train model on all dataset.")

    print("Answer is values of best parameters combination, and accuracy of last model of full dataset.")
    
    print("Always use random_state = work_id.")
    
    return(train.sample(450, random_state = work_id))
     
