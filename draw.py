import matplotlib.pyplot as plt
import re


def main():

    with open("result.txt") as f:
        lines = f.read().split("\n")

    # matplotlib

    # 每一行输出格式
    # sum_Episode: 133400 Episode:  1400 Score: 1832.1463695058753  Avg.Score: 3519.75, eps-greedy:  0.01 Time: 00:16:30 level:   10  num_success: 5  num_crash: 9  num_none_energy: 0  num_overstep: 1
    pattern = re.compile(r"\bEpisode:.*?(\d+).*?Avg\.Score:.*?(-?\d+\.?\d+)")
    avg_scores = []
    for line in lines:
        match_group = pattern.search(line)
        if match_group is None:
            continue
        episode, avg_score = match_group.group(1, 2)
        avg_scores.append(float(avg_score))

    plt.plot(avg_scores)
    plt.xlabel("episode / 10")
    # y 轴信息位于
    plt.ylabel("avg_score")

    plt.show()


if __name__ == "__main__":
    main()
