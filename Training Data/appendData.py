import csv

def process_text_file(input_file):
    data = []
    with open(input_file, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()
        for line in lines:
            print(line.strip().split("@")[0])
            print(line.strip().split("@")[1])
            sentence, sentiment = line.strip().split("@")
            data.append((sentence, sentiment))
    return data

def append_to_csv(output_file, data):
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

def main():
    input_file = 'Sentences_AllAgree.txt'
    output_file = 'data.csv'

    data = process_text_file(input_file)
    append_to_csv(output_file, data)

if __name__ == '__main__':
    main()
