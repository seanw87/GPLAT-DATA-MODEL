import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os

ip_range_country_province_path = "data/ip_range_overall_selected.csv"
output_path = "data/output"
output_file_pre = "ip_val_overall_selected_res_"

if __name__ == "__main__":
    result = []

    ip_pool = []

    df_iprange = pd.read_csv(ip_range_country_province_path, encoding='utf-8')
    print(df_iprange[0:10])

    output_file_no = 0
    for index, row in df_iprange.iterrows():
        minip = row['minip']
        maxip = row['maxip']
        print(index, minip, maxip, row['country'], row['province'], row['lngwgs'], row['latwgs'])

        # if row['country'] not in ('越南', '泰国', '中国', '新加坡', '印度尼西亚', '中国香港', '马来西亚', '巴西', '伊朗', '俄罗斯', '美国', '印度', '菲律宾'):
        #     print('非优先国家, pass')
        #     pass

        for ipval in range(minip, maxip+1):
            if output_file_no == 243:
                ipsingle = [ipval, minip, maxip, row['country'], row['province'], row['lngwgs'], row['latwgs']]
                result.append(ipsingle)

        if index % 30000 == 0:
            output_file_no += 1
            # df_res = pd.DataFrame(result,
            #                       columns=['ipval', 'minip', 'maxip', 'country', 'province', 'lngwgs', 'latwgs'])
            # df_res.to_csv(os.path.join(output_path, output_file_pre + str(output_file_no) + ".csv"), encoding='utf-8', index=None, mode='a')
            result = []

    df_res = pd.DataFrame(result, columns=['ipval', 'minip', 'maxip', 'country', 'province', 'lngwgs', 'latwgs'])
    df_res.to_csv(os.path.join(output_path, output_file_pre + str(output_file_no) + ".csv"), encoding='utf-8', index=None, mode='a')


    exit(0)
