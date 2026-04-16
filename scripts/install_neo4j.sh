#!/usr/bin/env bash
#
# Install Neo4j Community Edition 5.x on Debian/Ubuntu.
# Idempotent — safe to re-run.
#
# After install:
#   1. Change the default password with cypher-shell (see instructions printed at end)
#   2. Export NEO4J_PASSWORD before running seed_schema.py
#   Note: Community Edition only supports the default 'neo4j' and 'system' databases;
#   ForgeRAG uses the default 'neo4j' database.
#
# Reference: https://neo4j.com/docs/operations-manual/current/installation/linux/debian/
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[install_neo4j]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[install_neo4j]${NC} $*" >&2
}

err() {
    echo -e "${RED}[install_neo4j]${NC} $*" >&2
}

if [[ $EUID -eq 0 ]]; then
    err "Do not run as root. Script uses sudo where needed."
    exit 1
fi

# 1. Install prerequisites
log "Installing prerequisites..."
sudo apt-get update -qq
sudo apt-get install -y -qq wget gnupg lsb-release apt-transport-https ca-certificates

# 2. Add the Neo4j APT repo (official)
if [[ ! -f /etc/apt/sources.list.d/neo4j.list ]]; then
    log "Adding Neo4j APT repository..."
    wget -qO - https://debian.neo4j.com/neotechnology.gpg.key | \
        sudo gpg --dearmor -o /usr/share/keyrings/neo4j.gpg
    echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable 5" | \
        sudo tee /etc/apt/sources.list.d/neo4j.list > /dev/null
    sudo apt-get update -qq
else
    log "Neo4j APT repo already configured"
fi

# 3. Install Neo4j (Community edition is the default on the stable channel)
if ! dpkg -l | grep -q '^ii  neo4j '; then
    log "Installing Neo4j Community Edition..."
    sudo apt-get install -y neo4j
else
    log "Neo4j already installed: $(dpkg -s neo4j | awk '/^Version:/ {print $2}')"
fi

# 4. Harden config — bolt on localhost only, HTTP on localhost only
CONFIG=/etc/neo4j/neo4j.conf
if [[ -f "$CONFIG" ]]; then
    log "Locking listen addresses to localhost..."
    # Uncomment and set bolt listen address
    sudo sed -i -E 's|^#?server\.bolt\.listen_address=.*|server.bolt.listen_address=127.0.0.1:7687|' "$CONFIG"
    sudo sed -i -E 's|^#?server\.http\.listen_address=.*|server.http.listen_address=127.0.0.1:7474|' "$CONFIG"
    sudo sed -i -E 's|^#?server\.https\.enabled=.*|server.https.enabled=false|' "$CONFIG"
    # Leave default memory sizing — Neo4j auto-tunes from system memory
fi

# 5. Enable and start the service
log "Enabling and starting neo4j.service..."
sudo systemctl enable neo4j
sudo systemctl start neo4j

# 6. Health check
log "Waiting for Neo4j to respond on bolt://localhost:7687..."
for i in {1..30}; do
    if nc -z localhost 7687 2>/dev/null; then
        log "Neo4j is listening on bolt://localhost:7687"
        break
    fi
    sleep 2
    if [[ $i -eq 30 ]]; then
        err "Neo4j did not start within 60s. Check: sudo journalctl -u neo4j -e"
        exit 2
    fi
done

cat <<EOF

$(echo -e "${GREEN}Neo4j installation complete.${NC}")

Next steps (run these yourself in a terminal — they need interactive input):

1. Change the password from the default 'neo4j' to your own
   (Neo4j is already running, so we use ALTER USER rather than set-initial-password):

    $(echo -e "${YELLOW}cypher-shell -u neo4j -p neo4j -d system \\
        \"ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'YOUR_STRONG_PASSWORD'\"${NC}")

2. (No separate database to create — Neo4j Community Edition only supports the
   built-in 'neo4j' and 'system' databases. ForgeRAG uses the default 'neo4j' DB.)

3. Export the password for ForgeRAG to use (for current shell):

    $(echo -e "${YELLOW}export NEO4J_PASSWORD='YOUR_STRONG_PASSWORD'${NC}")

   To persist across reboots (for the systemd service):

    $(echo -e "${YELLOW}sudo mkdir -p /etc/forgerag${NC}")
    $(echo -e "${YELLOW}echo \"NEO4J_PASSWORD='YOUR_STRONG_PASSWORD'\" | sudo tee /etc/forgerag/env >/dev/null${NC}")
    $(echo -e "${YELLOW}sudo chmod 600 /etc/forgerag/env${NC}")

4. Seed the ForgeRAG schema:

    $(echo -e "${YELLOW}cd /home/nuc1/projects/ForgeRAG && ./venv/bin/python scripts/seed_schema.py${NC}")

EOF
