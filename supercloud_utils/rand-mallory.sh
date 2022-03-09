# This is a helper script that creates a $TMP_MALLORY_CONFIG
# and sets up http_proxy and https_proxy.

# From https://unix.stackexchange.com/a/423052/466333
OPEN_PORT_SMART=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
OPEN_PORT_NORMAL=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

# mktemp: https://unix.stackexchange.com/a/181938/466333
# trap: https://unix.stackexchange.com/a/181944/466333
export TMP_MALLORY_CONFIG=$(mktemp /tmp/mallory.XXXXXX)
trap 'rm -f -- "$TMP_MALLORY_CONFIG"' INT TERM HUP EXIT

cat > $TMP_MALLORY_CONFIG <<- EOM
{
  "id_rsa": "$HOME/.ssh/id_rsa",
  "local_smart": ":$OPEN_PORT_SMART",
  "local_normal": ":$OPEN_PORT_NORMAL",
  "remote": "ssh://$USER@login-3:22",
  "blocked": []
}
EOM

export http_proxy="http://localhost:$OPEN_PORT_NORMAL"
export https_proxy="http://localhost:$OPEN_PORT_NORMAL"

echo "Created temporary mallory config: $TMP_MALLORY_CONFIG"
# One should then run:
#     mallory -config $TMP_MALLORY_CONFIG
