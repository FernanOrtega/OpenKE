import os


def file_to_array(path: str) -> []:
    return [line.rstrip('\n') for line in open(path)][1:]


def array_to_file(array_to_save: [], array_path: str) -> None:
    with open(array_path, 'w') as a_file:
        a_file.write(str(len(array_to_save)) + "\n")
        for line in array_to_save:
            # print line
            a_file.write(line + "\n")


def merge_two_files(path1: str, path2: str, path_merge: str) -> []:
    result = file_to_array(path1)
    result.extend(file_to_array(path2))
    array_to_file(result, path_merge)

    return result


def merge_training_testing(dataset_path: str, source_name: str,
                           target_name: str) -> None:
    path1 = os.path.join(dataset_path, 'training', source_name)
    path2 = os.path.join(dataset_path, 'testing', source_name)
    path_merge = os.path.join(dataset_path, target_name)
    merge_two_files(path1, path2, path_merge)


def fix_dataset(dataset_path: str) -> None:
    merge_training_testing(dataset_path, 'clazzes.txt', 'clazzes.txt')
    merge_training_testing(dataset_path, 'entity2id.txt', 'entity2id.txt')
    merge_training_testing(dataset_path, 'relation2id.txt', 'relation2id.txt')
    merge_training_testing(dataset_path, 'train2id.txt', 'train2id.txt')


def main() -> None:
    root_path = 'KGERoller'
    for domain in os.listdir(root_path):        
        print("Merging for domain {}".format(domain))
        domain_path = os.path.join(root_path, domain)
        for site in os.listdir(domain_path):
            print("Merging for site {}".format(site))
            dataset_path = os.path.join(domain_path, site)
            fix_dataset(dataset_path)


if __name__ == "__main__":
    main()
