#!/bin/sh
# Creates the read-only account for HR admins' analytics questions, with no rights yet.
# MySQL runs this once, when the data volume is first initialised (as root).
#
# Its rights are granted by the set-up job (`hrm db grant-analytics`) after the migrations
# have created the views: MySQL refuses grants on views that do not exist yet.
set -eu

mysql --user=root --password="$MYSQL_ROOT_PASSWORD" <<SQL
CREATE USER IF NOT EXISTS 'hrm_analytics'@'%' IDENTIFIED BY '${MYSQL_ANALYTICS_PASSWORD}';
SQL
