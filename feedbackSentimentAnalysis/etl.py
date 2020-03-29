import warnings
warnings.filterwarnings("ignore")
import os
import xlrd
import jieba



jieba_dict_file = "data/jieba/dict.txt"
jieba.load_userdict(jieba_dict_file)

# 中文分词
def jieba_cut(sentence):
    res = jieba.cut(sentence, cut_all=False, HMM=True)
    res = "|".join(res)
    return str(res)




# 提取excel文件中的舆情内容（区分正负情绪）
def load_excel_data(excelFile):
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_name('总计')
    print("文件名称：" + excelFile)
    print("总行数：" + str(table.nrows))
    print("总列数：" + str(table.ncols))
    print("\n")

    posFeedbacks = []
    negFeedbacks = []

    for rowNum in range(table.nrows):
        row = table.row_values(rowNum)
        if row[4] == "中立":
            posFeedbacks.append(row[2])
        elif row[4] == "负面":
            negFeedbacks.append(row[2])

    return posFeedbacks, negFeedbacks

# 遍历data/feedbacks目录下所有excel文件，并提取舆情内容，落地至文件
def load_and_sink_total():
    excelFileDir = 'data/feedbacks/'
    excelFiles = os.listdir(excelFileDir)

    fPosFeedback = open("data/pos_feedbacks.txt", "w", encoding='UTF-8')
    fPosFeedbackSeg = open("data/pos_feedbacks_seg.txt", "w", encoding='UTF-8')
    fNegFeedback = open("data/neg_feedbacks.txt", "w", encoding='UTF-8')
    fNegFeedbackSeg = open("data/neg_feedbacks_seg.txt", "w", encoding='UTF-8')

    for excelFile in excelFiles:
        excelFilePath = excelFileDir + excelFile
        posFeedbacks, negFeedbacks = load_excel_data(excelFilePath)
        for line in posFeedbacks:
            line = line.strip().replace("|", ",")
            fPosFeedback.write(line + "\n")
            fPosFeedbackSeg.write(jieba_cut(line) + "\n")
        for line in negFeedbacks:
            line = line.strip().replace("|", ",")
            fNegFeedback.write(line + "\n")
            fNegFeedbackSeg.write(jieba_cut(line) + "\n")

    fPosFeedback.close()
    fPosFeedbackSeg.close()
    fNegFeedback.close()
    fNegFeedbackSeg.close()

# 遍历特定excel文件，并提取舆情内容，落地至文件
def load_and_sink(excelFile):
    excelFileDir = 'data/feedbacks/'

    fPosFeedback = open("data/pos_feedbacks.txt", "a", encoding='UTF-8')
    fPosFeedbackSeg = open("data/pos_feedbacks_seg.txt", "a", encoding='UTF-8')
    fNegFeedback = open("data/neg_feedbacks.txt", "a", encoding='UTF-8')
    fNegFeedbackSeg = open("data/neg_feedbacks_seg.txt", "a", encoding='UTF-8')

    excelFilePath = excelFileDir + excelFile
    posFeedbacks, negFeedbacks = load_excel_data(excelFilePath)
    for line in posFeedbacks:
        line = line.strip().replace("|", ",")
        fPosFeedback.write(line + "\n")
        fPosFeedbackSeg.write(jieba_cut(line) + "\n")
    for line in negFeedbacks:
        line = line.strip().replace("|", ",")
        fNegFeedback.write(line + "\n")
        fNegFeedbackSeg.write(jieba_cut(line) + "\n")

    fPosFeedback.close()
    fPosFeedbackSeg.close()
    fNegFeedback.close()
    fNegFeedbackSeg.close()




if __name__ == "__main__":
    load_and_sink_total()

