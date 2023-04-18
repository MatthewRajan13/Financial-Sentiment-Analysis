import csv

def process_csv_file(input_file, output_file):
    data = []
    with open(input_file, 'r', encoding='utf-8', errors='replace') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            lowercase_row = [cell.lower() for cell in row]
            data.append(lowercase_row)
            
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

def main():
    input_file = 'data.csv'
    output_file = 'Cleaned_Data.csv'

    process_csv_file(input_file, output_file)

if __name__ == '__main__':
    main()
