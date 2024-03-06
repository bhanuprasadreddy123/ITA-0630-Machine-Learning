import csv

def read_training_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data

def find_s_algorithm(training_data):
    hypothesis = ['0'] * (len(training_data[0]) - 1)
    
    for instance in training_data:
        if instance[-1] == 'Yes':
            for i in range(len(instance) - 1):
                if instance[i] != hypothesis[i] and hypothesis[i] == '0':
                    hypothesis[i] = instance[i]
                elif instance[i] != hypothesis[i]:
                    hypothesis[i] = '?'
    
    return hypothesis

def print_hypothesis(hypothesis):
    print("The most specific hypothesis is:", hypothesis)

def main():
    file_path = 'enjoysport.csv'
    training_data = read_training_data(file_path)
    
    hypothesis = find_s_algorithm(training_data)
    
    print_hypothesis(hypothesis)

if __name__ == "__main__":
    main()
