import csv

def find_s(training_data):
    hypothesis = ['0'] * (len(training_data[0]) - 1)
    for instance in training_data:
        if instance[-1] == 'Yes':  
            for i in range(len(instance) - 1):  
                if hypothesis[i] == '0': 
                    hypothesis[i] = instance[i] 
                elif hypothesis[i] != instance[i]:  
                    hypothesis[i] = '?'  
    return hypothesis

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def main():
    training_data = read_csv('enjoysport.csv')
    hypothesis = find_s(training_data)
    print("The most specific hypothesis is:", hypothesis)

if __name__ == "__main__":
    main()
