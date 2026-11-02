
import pandas as pd

df = pd.read_csv("data/priority_labeling.csv")

priority_labels = {
    # Tickets 1–10
    1656: "Low",
    855: "Low",
    13767: "Medium",
    2931: "Low",
    4622: "Medium",
    4247: "Low",
    1257: "Medium",
    2781: "Medium",
    1205: "Low",
    887: "Low",

    # Tickets 11–20
    10474: "Low",
    4539: "Low",
    10830: "Medium",
    905: "Low",
    602: "Medium",
    9263: "Low",
    2265: "Low",
    1031: "Medium",
    10996: "Medium",
    794: "Medium",

    # Tickets 21–30
    695: "Low",
    354: "Low",
    1237: "Medium",
    1068: "Medium",
    6341: "Low",
    8741: "Medium",
    4572: "Low",
    1330: "Medium",
    837: "Low",
    7468: "Low",

    # Tickets 31–40
    3698: "Low",
    1089: "Low",
    591: "Low",
    941: "Medium",
    4289: "Medium",
    4462: "Low",
    6192: "Medium",
    1004: "Low",
    1696: "Low",
    11462: "Low",

    # Tickets 41–60
    2132: "Low",
    3019: "Low",
    5330: "Low",
    7690: "Medium",
    1296: "Medium",
    6441: "Low",
    1067: "Low",
    9887: "Medium",
    789: "Low",
    2164: "High",
    3456: "Medium",
    7379: "Low",
    240: "Low",
    785: "Low",
    963: "Low",
    1267: "Low",
    1414: "Low",
    13809: "Low",
    341: "Medium",
    1158: "Medium",

    # Tickets 61–80
    1264: "Medium",
    1866: "Low",
    564: "Medium",
    709: "Low",
    1057: "Medium",
    5009: "Low",
    2368: "Medium",
    7375: "Low",
    75: "Medium",
    1293: "Low",
    7787: "Low",
    1064: "Medium",
    3692: "Low",
    8758: "Medium",
    1286: "Low",
    4530: "Low",
    1135: "Medium",
    7943: "Low",
    1995: "Medium",
    13763: "Low",

    # Tickets 81–100
    896: "Medium",
    9986: "Medium",
    10968: "Medium",
    9585: "Low",
    4285: "Low",
    10824: "Low",
    7573: "High",
    1732: "Low",
    7165: "Low",
    7480: "Low",
    6398: "Low",
    916: "Low",
    7160: "Low",
    6258: "Medium",
    4673: "Medium",
    1149: "Medium",
    836: "Medium",
    1235: "Low",
    1409: "Low",
    10280: "Medium",

    # Tickets 101–120
    991: "Medium",
    6955: "Low",
    11074: "Medium",
    10454: "Low",
    378: "Low",
    10535: "Medium",
    1476: "Low",
    4800: "Low",
    3678: "Low",
    1143: "Medium",
    903: "Medium",
    1077: "Medium",
    14302: "Medium",
    842: "Medium",
    7098: "Low",
    2242: "Low",
    894: "Medium",
    10060: "Low",
    7894: "Medium",
    5925: "Medium",

    # Tickets 121–141
    945: "Low",
    1027: "Low",
    2211: "Low",
    9053: "Low",
    648: "Medium",
    1096: "Medium",
    5332: "Low",
    1411: "Low",
    9791: "Low",
    1311: "Medium",
    7896: "Low",
    588: "Low",
    1297: "Low",
    11138: "Low",
    9559: "Medium",
    1005: "Medium",
    10406: "Medium",
    1642: "Low",
    563: "Low",
    909: "Medium",
    953: "Medium"
}

new_labels = df["id"].map(priority_labels)
df["priority"] = new_labels.where(new_labels.notna(), df["priority"])

df.to_csv("data/priority_labeling.csv", index=False)

print("Priority labels updated successfully!")

print("\nPriority distribution:")
print(df["priority"].value_counts(dropna=False))

print("\nTotal labelled:", df["priority"].notna().sum())
print("Remaining:", df["priority"].isna().sum())
