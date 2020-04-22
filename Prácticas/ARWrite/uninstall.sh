#!/bin/bash
(
# =================================================================
echo "# Removing files" ;
pkexec sudo rm -rf /opt/arwrite;
sleep 0.5;
# =================================================================
echo "75"
echo "# Removing desktop entry" ;
sudo rm -f /usr/share/applications/ARWrite.desktop
sleep 0.5;
# =================================================================
echo "100"
echo "# ARWrite has been successfully uninstalled." ;
) |
zenity --progress \
  --title="ARWrite uninstaller" \
  --percentage=0 \
  --no-cancel

(( $? != 0 )) && zenity --error --text="Error uninstalling ARWrite."

exit 0