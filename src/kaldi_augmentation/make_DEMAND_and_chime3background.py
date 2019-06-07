#!/usr/bin/env python
# python >= 2.7 required
import os, errno, fnmatch


# https://stackoverflow.com/a/600612/3342981
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# https://stackoverflow.com/a/2186565/3342981
def get_matchingfiles_recursive(src, pattern="*.wav"):
    matches = []
    for root, dirnames, filenames in os.walk(src):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


def read_file(filepath, asdict=True):
    """
        read a file CONTAINING DATA IN TWO COLUMNS as dictionary
    """
    with open(filepath, "rt") as fid_read:
        dictionary = dict([line.split() for line in fid_read])
    return dictionary


def filterkeysfromdict(dictionary, pattern):
    output = []
    for k,v in dictionary.items():
        if pattern in k:
            output.append(k)
    return output


if __name__ == "__main__":
    dir_DEMAND="/export/corpora/DEMAND"
    dir_chime3background="/export/corpora4/CHiME4/CHiME3/data/audio/16kHz/backgrounds"
    dir_target = "data"     # change this to "." if you want folder to be made in working directory

    mkdir_p(os.path.join(dir_target, "demand_train"))
    mkdir_p(os.path.join(dir_target, "chime3background"))
    mkdir_p(os.path.join(dir_target, "chime3background_train"))
    mkdir_p(os.path.join(dir_target, "chime3background_eval"))

    # DEMAND
    utts = get_matchingfiles_recursive(dir_DEMAND)
    wavscp = os.path.join(dir_target, "demand_train", "wav.scp")
    utt2spk = os.path.join(dir_target, "demand_train", "utt2spk")
    spk2utt = os.path.join(dir_target, "demand_train", "spk2utt")
    with open(wavscp, "wt") as fw, open(utt2spk, "wt") as fw1, open(spk2utt, "wt") as fw2:
        for utt in utts:
            if "ch01" in utt:
                identifier = utt.split("/")[4] + "_ch01"
                line = identifier + " " + utt + "\n"
                fw.write(line)
                line = identifier + " " + identifier + "\n"
                fw1.write(line)
                fw2.write(line)
    print("written: " + os.path.abspath(wavscp) + "\n")
    print("written: " + os.path.abspath(utt2spk) + "\n")
    print("written: " + os.path.abspath(spk2utt) + "\n")

    # chime3background
    utts = get_matchingfiles_recursive(dir_chime3background)
    wavscp = os.path.join(dir_target, "chime3background", "wav.scp")
    utt2spk = os.path.join(dir_target, "chime3background", "utt2spk")
    spk2utt = os.path.join(dir_target, "chime3background", "spk2utt")
    with open(wavscp, "wt") as fw, open(utt2spk, "wt") as fw1, open(spk2utt, "wt") as fw2:
        for utt in utts:
            if ".CH1." in utt:
                identifier = "backgrounds_" + utt.split("/")[-1].split(".")[:-2][0]
                line = identifier + " " + utt + "\n"
                fw.write(line)
                line = identifier + " " + identifier + "\n"
                fw1.write(line)
                fw2.write(line)
    print("written: " + os.path.abspath(wavscp) + "\n")
    print("written: " + os.path.abspath(utt2spk) + "\n")
    print("written: " + os.path.abspath(spk2utt) + "\n")

    # get keys for train and test of chime3background
    dictionary = read_file(wavscp)
    envs = []
    for k,v in dictionary.items():
        envs.append(k.split("_")[-1])
    envs_all = envs
    envs = list(set(envs))
    keys_train = []
    keys_test = []
    for env in envs:
        count = sum([1 for _ in envs_all if _ == env])
        keys_curr = filterkeysfromdict(dictionary, env)
        assert len(keys_curr) != 0
        if len(keys_curr) == 1:
            keys_train += keys_curr
        elif len(keys_curr) == 2:
            keys_train += [keys_curr[0]]
            keys_test += [keys_curr[-1]]
        elif len(keys_curr) == 3:
            keys_train += keys_curr[:2]
            keys_test += [keys_curr[-1]]
        elif len(keys_curr) == 4:
            keys_train += keys_curr[:3]
            keys_test += [keys_curr[-1]]
        else:
            keys_train += keys_curr[:3]
            keys_test += keys_curr[3:]

    # chime3background_train
    wavscp = os.path.join(dir_target, "chime3background_train", "wav.scp")
    utt2spk = os.path.join(dir_target, "chime3background_train", "utt2spk")
    spk2utt = os.path.join(dir_target, "chime3background_train", "spk2utt")
    with open(wavscp, "wt") as fw, open(utt2spk, "wt") as fw1, open(spk2utt, "wt") as fw2:
        for k in keys_train:
            line = k + " " + dictionary[k] + "\n"
            fw.write(line)
            line = k + " " + k + "\n"
            fw1.write(line)
            fw2.write(line)
    print("written: " + os.path.abspath(wavscp) + "\n")
    print("written: " + os.path.abspath(utt2spk) + "\n")
    print("written: " + os.path.abspath(spk2utt) + "\n")

    # chime3background_test
    wavscp = os.path.join(dir_target, "chime3background_eval", "wav.scp")
    utt2spk = os.path.join(dir_target, "chime3background_eval", "utt2spk")
    spk2utt = os.path.join(dir_target, "chime3background_eval", "spk2utt")
    with open(wavscp, "wt") as fw, open(utt2spk, "wt") as fw1, open(spk2utt, "wt") as fw2:
        for k in keys_test:
            line = k + " " + dictionary[k] + "\n"
            fw.write(line)
            line = k + " " + k + "\n"
            fw1.write(line)
            fw2.write(line)
    print("written: " + os.path.abspath(wavscp) + "\n")
    print("written: " + os.path.abspath(utt2spk) + "\n")
    print("written: " + os.path.abspath(spk2utt) + "\n")
