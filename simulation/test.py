import io,sys
import operator

fin_locate_test_set = io.open(sys.argv[2],"r",encoding="utf-8")

line = fin_locate_test_set.readline()

user_influenced_users = {}

while line:
    line = line.rstrip()
    test_case = line.split(",")
    key = test_case[0]

    influenced_users_time = test_case[3].split(";")
    influenced_users = []
    for item in influenced_users_time:
        influenced_users.append(item.split(":")[0])

    if user_influenced_users.has_key(key):
        for item in influenced_users:
            user_influenced_users.get(key).add(item)
    else:
        user_influenced_users[key] = set(influenced_users)

    line = fin_locate_test_set.readline()

user_influence_count = {}

for k,v in user_influenced_users.iteritems():
    user_influence_count[k] = len(v)

sorted_users = sorted(user_influence_count.items(), key=operator.itemgetter(0))
# print sorted_users
final_users_set = []

for k,v in sorted_users:
    if(len(final_users_set) <= 500):
        final_users_set.append(k)
        for user in user_influenced_users[k]:
            final_users_set.append(user)
    else:
        break

# print "final users :"
#
# print final_users_set

#read checkins and write checkins for sampled users
fin_checkins = io.open(sys.argv[1],"r",encoding="utf-8")
fout_checkins = io.open(sys.argv[1]+"500users","w",encoding="utf-8")

line = fin_checkins.readline()

user = ""
prev_user = ""
flag = 0

while line:
    prev_user = user
    line = line.rstrip()
    checkin = line.split(",")
    user = checkin[0]
    if prev_user != user:

        if user in final_users_set:
            fout_checkins.write(line+"\n")
            flag = 1
        else:
            flag = 0
    elif prev_user == user and flag == 1:
        fout_checkins.write(line + "\n")
    else:
        pass
    line = fin_checkins.readline()

fin_checkins.close()
fout_checkins.close()