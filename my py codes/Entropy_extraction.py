"""
Renyi Entropy:
    arguments: data <pandas dataframe>, alpha (parameter) = <int> 2
    output: Renyi Entropy of the signal
"""

def renyi_entropy(data,alpha):

    #make sure to round the signal points to integer
    data=data.astype('int64')
    #iterate each signal
    for i in range(<>):
        X=data[str(i)].values
        data_set = list(set(X))
        freq_list = []
        for entry in data_set:
            counter = 0.
            for i in X:
                if i == entry:
                    counter += 1
            freq_list.append(float(counter)/len(X))
        summation=0
        for freq in freq_list:
            summation+=math.pow(freq,alpha)
        Renyi_En=(1/float(1-alpha))*(math.log(summation,2))

"""
For Tsallis Entropy only change is the last line of Renyi Ent.
instead of
<Renyi_En=(1/float(1-alpha))*(math.log(summation,2))>

replace it with,

<Tsa_En=(1/float(alpha-1))*(1-summation)>
"""
