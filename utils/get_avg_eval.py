import pandas


def read_csv(file_name):
    csv = pandas.read_csv(file_name)
    actions = csv['action']
    cols = csv.columns[1:]
    res = {}
    for i, action in enumerate(actions):
        res[action] = []
        for col in cols:
            res[action].append(csv[col][i])
    return res


def add_dict_b_to_a(dict_a, dict_b):
    for k, v in dict_b.items():
        # print(k, v)
        assert len(dict_a[k]) == len(dict_b[k])
        for i in range(len(dict_b[k])):
            dict_a[k][i] += dict_b[k][i]


def divide_dict_by_num(dct, num):
    for k, v in dct.items():
        for i in range(len(dct[k])):
            dct[k][i] /= num


def write_dict_to_csv(dct, file_name):
    actions = list(dct.keys())
    # print(actions)
    res_dct = {'actions': actions}
    frame_tot = len(dct[actions[0]])
    for i in range(1, frame_tot + 1):
        timestap = '%dms' % (i * 40)
        all_action_res_at_time_t = []
        for action in actions:
            all_action_res_at_time_t.append(dct[action][i - 1])
        res_dct[timestap] = all_action_res_at_time_t
    res_dct = pandas.DataFrame(res_dct)
    res_dct.to_csv(file_name, index=False)


if __name__ == "__main__":
    import os

    for root_dir in os.listdir('./log_tb'):
        avg_res = None

        csv_path = f"./log_tb/{root_dir}/csv/test_model_%04d_all_eval.csv"

        ok = True
        for i in range(191, 200 + 1):
            if not os.path.exists(csv_path % i):
                ok = False
                break
        if not ok:
            continue
        print(root_dir, ok)
        for i in range(191, 200 + 1):
            csv = csv_path % i
            res_dict = read_csv(csv)
            if avg_res is None:
                avg_res = res_dict
            else:
                add_dict_b_to_a(avg_res, res_dict)

        divide_dict_by_num(avg_res, 10)
        write_dict_to_csv(avg_res, f"./log_tb/{root_dir}/csv/avg_eval.csv")
