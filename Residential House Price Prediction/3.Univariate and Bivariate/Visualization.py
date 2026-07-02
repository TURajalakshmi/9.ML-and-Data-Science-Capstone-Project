import matplotlib.pyplot as plt
import seaborn as sns


class visualization():

    # 1. Histogram
    def Histogram(dataset, column):

        plt.figure(figsize=(8,5))

        sns.histplot(
            dataset[column],
            kde=True
        )

        plt.title(column + " Distribution")

        plt.show()


    # 2. Boxplot
    def BoxPlot(dataset, column):

        plt.figure(figsize=(8,5))

        sns.boxplot(
            x=dataset[column]
        )

        plt.title(column + " Boxplot")

        plt.show()


    # 3. Countplot
    def CountPlot(dataset, column):

        plt.figure(figsize=(10,5))

        sns.countplot(
            x=dataset[column]
        )

        plt.xticks(rotation=45)

        plt.title(column + " Count Plot")

        plt.show()


    # 4. Scatter Plot
    def Scatter(dataset, x, y):

        plt.figure(figsize=(8,5))

        sns.scatterplot(
            data=dataset,
            x=x,
            y=y
        )

        plt.title(x + " vs " + y)

        plt.show()


    # 5. Correlation Heatmap
    def Heatmap(dataset):

        plt.figure(figsize=(14,10))

        sns.heatmap(
            dataset.corr(
                numeric_only=True
            ),
            annot=False,
            cmap="coolwarm"
        )

        plt.title("Correlation Heatmap")

        plt.show()


    # 6. Pairplot
    def PairPlot(dataset, columns):

        sns.pairplot(
            dataset[columns]
        )

        plt.show()


    # 7. Violin Plot
    def Violin(dataset, column):

        plt.figure(figsize=(8,5))

        sns.violinplot(
            y=dataset[column]
        )

        plt.title(column + " Violin Plot")

        plt.show()


    # 8. Line Plot
    def Line(dataset, x, y):

        plt.figure(figsize=(8,5))

        sns.lineplot(
            data=dataset,
            x=x,
            y=y
        )

        plt.title(x + " vs " + y)

        plt.show()

    # 9. joint Plot
    def JointPlot(dataset, x, y):
        
        sns.jointplot(
            data=dataset,
            x=x,
            y=y,
            kind="hex"
        )
        
        plt.show()

    # 10. Dist Plot
    def DistPlot(dataset,column):

        plt.figure(figsize=(8,5))

        sns.histplot(
            dataset[column],
            kde=True
        )

        plt.title(column)

        plt.show()

        