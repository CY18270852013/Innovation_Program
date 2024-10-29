import seaborn as sns
import matplotlib.pyplot as plt
iris_df = sns.load_dataset('iris')
sns.pairplot(iris_df, hue = 'species')
