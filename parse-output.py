import argparse

def main():
    parser = argparse.ArgumentParser(description='Output Parser')
    parser.add_argument('--output_path', type=str, help='path to output file.')
    parser.add_argument('--results_path', type=str, help='path to results file.')
    parser.add_argument('--header', type=str, help='option to write columns in header of csv.')
    args = parser.parse_args()

    with open(args.results_path, 'a') as fd:
        if args.header == "cla":
            fd.write("model,k_shots,num_classes,num_filters,num_conv_blocks,dataset,split,epoch,mix,accuracy\n")
        with open(args.output_path, 'r') as output:
            for line in output:
                if "RunResults:" in line:
                    fd.write(line[len("RunResults:"):]) 
if __name__ == '__main__':
    main()
