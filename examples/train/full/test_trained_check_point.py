import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference', type=str, required=True)

    return parser.parse_args()






if __name__ == '__main__':
    args = parse_args()
    if args.inference == 'infer':
        print(args)