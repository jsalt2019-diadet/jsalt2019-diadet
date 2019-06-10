BEGIN{
  while(getline < u2u)
  {
    u[$1]=$2;
  }
}
$0 !~ "--snrs" { print $1,u[$1],"None","40"}
$0 ~ "--snrs" {
  for(i=2;i<=NF;i++)
  {
    if($i ~ "--snrs")
    {
      sub(/--snrs=./,"",$i);
      sub(/.$/,"",$i);
      n_snrs=split($i,snrs,",");
      avg_snr=0
      for(j=1;j<=n_snrs;j++)
      {
        if (aug=="babble")
        {
          avg_snr+=10^(-snrs[j]/10)
        }
        else
        {
          avg_snr+=snrs[j]
        }
      }
      if (aug=="babble")
      {
        avg_snr=10*log(1/avg_snr)/log(10)
      }
      else
      {
        avg_snr/=n_snrs
      }
      break
    }
  }
  print $1,u[$1],aug,avg_snr }


