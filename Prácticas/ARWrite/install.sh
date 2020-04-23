#!/bin/bash
if zenity --question --text="Do you want to install ARWrite?"
then
(
# =================================================================
echo "# Installing python packages" ;
pkexec sudo apt-get install -y python3 python3-pip python3-tk;

# =================================================================
echo "25"
echo "# Installing python libraries" ;
pip3 install -r requirements.txt;
# =================================================================
echo "50"
echo "# Copying files" ;
mkdir -p ~/.arwrite;
cp -R ./ ~/.arwrite;
sleep 0.5;
# =================================================================
echo "75"
echo "# Creating desktop entry" ;
cp -f ARWrite.desktop ~/.local/share/applications/
sed -ie "s|HOME|$HOME|g" "$HOME/.local/share/applications/ARWrite.desktop"
sleep 0.5;
# =================================================================
echo "100"
echo "# ARWrite has been successfully installed." ;
) |
zenity --progress \
  --title="ARWrite installer" \
  --percentage=0 \
  --no-cancel

(( $? != 0 )) && zenity --error --text="Error installing ARWrite."

exit 0
fi
