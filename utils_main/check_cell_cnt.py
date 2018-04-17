import os

train_dir = '/data/public/rw/datasets/dsb2018/external_data/crowd_sourced/train'
csv_dir = '/data/public/rw/datasets/dsb2018/external_data/crowd_sourced/PSB_2015_ImageSize_400/Nuclei_Detection/AutomatedMethod'


def main():
    cell_cnt_map = get_cell_cnt_map()

    for dir in os.listdir(train_dir):
        dir_name = os.path.join(train_dir, dir, 'masks')
        if os.path.exists(dir_name):
            cell_cnt = len(os.listdir(dir_name))
            if dir in cell_cnt_map:
                if cell_cnt_map[dir] == cell_cnt:
                    pass
                    # print('success -', dir)
                else:
                    print('failed1 -', dir, cell_cnt_map[dir], cell_cnt)
            else:
                print('failed2 -', dir)
        else:
            print('not exists directory -', dir_name)


def get_cell_cnt_map():
    cell_cnt_map = {}
    for file in os.listdir(csv_dir):
        file_name = os.path.join(csv_dir, file)
        num_lines = sum(1 for _ in open(file_name))
        cell_cnt_map[os.path.splitext(file)[0]] = num_lines

    return cell_cnt_map


if __name__ == '__main__':
    main()
