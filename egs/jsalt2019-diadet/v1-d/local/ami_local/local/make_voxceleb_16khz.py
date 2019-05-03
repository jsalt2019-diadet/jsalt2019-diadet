import sys, os

src_dir = sys.argv[1]
data_dir = sys.argv[2]
num_missing = 0
num_found = 0
if not os.path.exists(data_dir):
  os.makedirs(data_dir)
wavscp = open(data_dir + "/wav.scp" , 'w')
utt2spk = open(data_dir + "/utt2spk" , 'w')
for subdir, dirs, files in os.walk(src_dir):
  for file in files:
    filename = os.path.join(subdir, file)
    if filename.endswith(".txt"):
      lines = open(filename, 'r').readlines()
      if "POI" in lines[0]:
        spkr = lines[0].rstrip().split()[1].replace(".","")
        ytid = lines[1].rstrip().split()[2]
        dur = int(float(lines[2].rstrip().split()[2]) * 100)
        utt_id = spkr + "-" + ytid
        flac_path = src_dir + "/" + ytid + ".flac"
        wav  = utt_id + " sox -t flac " + src_dir + "/" + ytid + ".flac -t wav -r 16k -b 16 - channels 1 |\n"
        if os.path.isfile(flac_path):
          wavscp.write(wav)
          utt2spk.write(utt_id + " " + spkr + "\n")
          num_found += 1
        else:
          print "File " + flac_path + " doesn't exist"
          num_missing += 1
print "Missing " + str(num_missing) + " files out of " + str(num_missing + num_found)




