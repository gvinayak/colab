import io
import sys

#read ckeckins file and store w.r.t. each user
def importCheckins(filename):
    fin_checkins = io.open(filename,"r",encoding="utf-8")

    line = fin_checkins.readline()

    checkins = {}

    while line:
        line = line.rstrip()
        checkin = line.split(",")
        pair = (checkin[1], checkin[2])
        if checkins.has_key(checkin[0]):
            checkins.get(checkin[0]).append(pair)
        else:
            checkins[checkin[0]] = [pair]

        line = fin_checkins.readline()

    return checkins;


#read venues file and store venue info
def importVenues(filename):
    fin_venues = io.open(filename,"r",encoding="utf-8")

    line = fin_venues.readline()

    venues = {}

    while line:
        line = line.rstrip()
        venue_info = line.split(",")
        geo_cordinates = (venue_info[1],venue_info[2])
        pair = (geo_cordinates,venue_info[3])
        if venues.has_key(venue_info[0]):
            venues.get(venue_info[0]).append(pair)
        else:
            venues[venue_info[0]] = [pair]

        line = fin_venues.readline()

    return venues;

#read social connections and store in map
def importConnections(filename):
    fin_social = io.open(filename,"r",encoding="utf-8")

    line = fin_social.readline()

    connections = {}

    while line:
        line = line.rstrip()
        social_info = line.split(",")
        if connections.has_key(social_info[0]):
            connections.get(social_info[0]).append(social_info[1])
        else:
            connections[social_info[0]] = [social_info[1]]

        line = fin_social.readline()

    return connections;