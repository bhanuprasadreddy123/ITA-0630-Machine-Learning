import csv

def read_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        return [row for row in reader]

def is_consistent(instance, hypothesis):
    return all(h == '?' or h == val for h, val in zip(hypothesis, instance))

def candidate_elimination(training_data):
    num_attributes = len(training_data[0]) - 1
    S = [['0'] * num_attributes]
    G = [['?'] * num_attributes]
    
    for instance in training_data:
        if instance[-1] == 'Yes':
            S = [h for h in S if is_consistent(instance, h)]
            G = [h for h in G if any(is_consistent(instance, h) for h in S)]
            G = [g for g in G if not any(is_consistent(instance, g) for g in S)]
        else:
            S = [generalize(h, instance) for h in S if not is_consistent(instance, h)]
            G = [g for g in G if is_consistent(instance, g)]

    return S, G

def generalize(hypothesis, instance):
    return [val if hyp == '0' else '?' for hyp, val in zip(hypothesis, instance)]

def main():
    training_data = read_csv('enjoysport.csv')
    S, G = candidate_elimination(training_data)
    
    print("Final hypothesis S:")
    for hypothesis in S:
        print(hypothesis)
    
    print("\nFinal hypothesis G:")
    for hypothesis in G:
        print(hypothesis)

if __name__ == "__main__":
    main()
