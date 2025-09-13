#!/bin/bash
 
function scan_wifi() {
    local bandwidth=$1
    local channel=$2
    local MAC_cli=$3

    local chanSpec=$(mcp -C 1 -N 1 -c "$channel/$bandwidth" -m $MAC_cli)
    # Especificando o endereço MAC, se está indicando de qual dispositivos se vão coletar os pacotes com os dados CSI, se o roteador ou o cliente que está fazendo o ping. 
 
    pkill wpa_supplicant
    ifconfig wlan0 up
 
    nexutil -Iwlan0 -s500 -b -l34 "-v$chanSpec"
 
    # setting up monitor fails when it already exists. Can be happily ignored.
    iw phy `iw dev wlan0 info | gawk '/wiphy/ {printf "phy" $2}'` interface add mon0 type monitor 2> /dev/null
    ifconfig mon0 up
}

sleep 5

codigo_part=$1
codigo_atv=$2
bandwidth=$3
channel=$4
MAC_cli=$5
nPackets=$6

NOW=$(date "+%Y_%m_%d_-_%H_%M_%S")
FILE=${codigo_atv}_${NOW}_bw_${bandwidth}_ch_${channel}.pcap

mkdir ./scans/
mkdir ./scans/"$codigo_part"
touch ./scans/"$codigo_part"/"$FILE"

date
scan_wifi "$bandwidth" "$channel" "$MAC_cli"

tcpdump -i wlan0 dst port 5500 -c "$nPackets" -w ./scans/"$codigo_part"/"$FILE"

date
#sshpass -p "password" scp ./scans/"$codigo_part"/"$FILE" user@192.168.0.2:/home/midiacom/CSI/scans/"$codigo_part"/"$FILE"
