import pandas as pd
import numpy as np

class Analysis():
    
    def QuanQual(dataset):
        qual=[]
        quan=[]
        for columnName in dataset.columns:
            if(dataset[columnName].dtypes=='O'):
                qual.append(columnName)   
            else:
                quan.append(columnName)
        return qual,quan

    def Univariate(dataset,quan):
        descriptive=pd.DataFrame(index=["Mean","Median","Mode","Q1:25%","Q2:50%","Q3:75%","99%","Q4:100%","IQR",
                                             "1.5rule","Var","Std","kurtosis","skew"],columns=quan)
        for columnName in quan:
            descriptive[columnName]["Mean"]=dataset[columnName].mean()
            descriptive[columnName]["Median"]=dataset[columnName].median()
            descriptive[columnName]["Mode"]=dataset[columnName].mode()[0]
            descriptive[columnName]["Q1:25%"] = dataset[columnName].quantile(0.25)
            descriptive[columnName]["Q2:50%"] = dataset[columnName].quantile(0.50)
            descriptive[columnName]["Q3:75%"] = dataset[columnName].quantile(0.75)
            descriptive[columnName]["99%"]=np.percentile(dataset[columnName],99)
            descriptive[columnName]["Q4:100%"]=dataset.describe()[columnName]["max"]
            descriptive[columnName]["IQR"]=descriptive[columnName]["Q3:75%"]-descriptive[columnName]["Q1:25%"]
            descriptive[columnName]["1.5rule"]=1.5*descriptive[columnName]["IQR"]
            descriptive[columnName]["Var"]=dataset[columnName].var()
            descriptive[columnName]["Std"]=dataset[columnName].std()
            descriptive[columnName]["kurtosis"]=dataset[columnName].kurtosis()
            descriptive[columnName]["skew"]=dataset[columnName].skew()
        return descriptive    
    
    
    def Outlier_Table(dataset, quan):
        descriptive =pd.DataFrame(index=["Q1:25%","Q3:75%","IQR","1.5rule","Lesser","Greater","Min","Max",],columns=quan)
        for columnName in quan:
            descriptive[columnName]["Q1:25%"] = dataset[columnName].quantile(0.25)
            descriptive[columnName]["Q3:75%"] = dataset[columnName].quantile(0.75)
            descriptive[columnName]["IQR"]=descriptive[columnName]["Q3:75%"]-descriptive[columnName]["Q1:25%"]
            descriptive[columnName]["1.5rule"]=1.5*descriptive[columnName]["IQR"]
            descriptive[columnName]["Lesser"]=descriptive[columnName]["Q1:25%"]-descriptive[columnName]["1.5rule"]
            descriptive[columnName]["Greater"]=descriptive[columnName]["Q3:75%"]+descriptive[columnName]["1.5rule"]
            descriptive[columnName]["Min"]=dataset[columnName].min()
            descriptive[columnName]["Max"]=dataset[columnName].max()
        return descriptive
    
    def Find_Outliers(quan, descriptive):
        lesser=[]
        greater=[]

        for columnName in quan:
            if descriptive[columnName]["Min"]<descriptive[columnName]["Lesser"]:
                lesser.append(columnName)
            if descriptive[columnName]["Max"]>descriptive[columnName]["Greater"]:
                greater.append(columnName)
        return lesser,greater

    def Replace_Outlier(dataset,descriptive,lesser,greater):
        for columnName in lesser:
            dataset.loc[dataset[columnName] < descriptive[columnName]["Lesser"],columnName] = descriptive[columnName]["Lesser"]
        for columnName in greater:
            dataset.loc[dataset[columnName] > descriptive[columnName]["Greater"],columnName] = descriptive[columnName]["Greater"]
        return dataset

    def freqTable(columnName,dataset):
        freqTable=pd.DataFrame(columns=["Unique_values","Frequency","Relative_Frequency","Cumsum"])
        freqTable["Unique_values"]=dataset[columnName].value_counts().index
        freqTable["Frequency"]=dataset[columnName].value_counts().values
        freqTable["Relative_Frequency"] = (freqTable["Frequency"] / len(dataset))
        freqTable["Cumsum"]=freqTable["Relative_Frequency"].cumsum()
        return freqTable

    def get_pdf_probability(dataset,startrange,endrange):
        from matplotlib import pyplot
        from scipy.stats import norm
        import seaborn as sns
        ax = sns.histplot(dataset, kde=True)
        pyplot.axvline(startrange,color='Red')
        pyplot.axvline(endrange,color='Red')
        sample = dataset
        sample_mean =sample.mean()
        sample_std = sample.std()
        print('Mean=%.3f, Standard Deviation=%.3f' % (sample_mean, sample_std))
        dist = norm(sample_mean, sample_std)
        values = [value for value in range(startrange, endrange)]
        probabilities = [dist.pdf(value) for value in values]    
        prob=sum(probabilities)
        print("The area between range({},{}):{}".format(startrange,endrange,sum(probabilities)))
        return prob 