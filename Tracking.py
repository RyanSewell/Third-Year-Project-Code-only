import numpy as np


#floorPoints = [[100,100],[200,200]]
#tracks = []

def objects(floorPoints, tracks):
    
    predictedPos = []
    posMaxDist = 90
    maxUnupdated = 5

    #first 0 how many since last proper update, second updated this turn, third if finished
    if len(tracks) < 1:
        for i in range(len(floorPoints)):
            tracks.append(([floorPoints[i]],[0, 0, 0]))
            #print(tracks)
    else:
        for i in range(len(tracks)):
            if tracks[i][1][2] == 0:
                if tracks[i][1][0] < maxUnupdated:
                    if len(floorPoints) > 0:
                        dist = []
                        if len(tracks[i][0]) < 2:
                            if len(floorPoints) > 0:
                                for j in range(len(floorPoints)):
                                    dist.append(np.linalg.norm(tracks[i][0][0] - floorPoints[j]))

                        else:
                            predictedPos.append(tracks[i][0][0] - tracks[i][0][1] + tracks[i][0][0])
                            if len(floorPoints) > 0:
                                for j in range(len(floorPoints)):
                                    dist.append(np.linalg.norm(predictedPos[len(predictedPos)-1] - floorPoints[j]))
            
                        lowestIndex = dist.index(min(dist))
                        trackCopy = tracks[i][0][0][:]
                        floorCopy = floorPoints[:]

                        if dist[lowestIndex] < posMaxDist:
                            tracks[i][0].insert(0, floorCopy[lowestIndex])
                            floorPoints = np.delete(floorPoints, lowestIndex, 0)
                            #0 1 as a an updated has happened so reset count of not happened and 
                            tracks[i][1][0] = 0
                            tracks[i][1][1] = 1

                        else:
                            #add 1 as not updated this time but carry on till value to deleted, 0 to to say no updated this frame so use anotehr colour
                            if len(predictedPos) > 1:
                                tracks[i][0].insert(0, predictedPos[len(predictedPos)-1])
                                tracks[i][1][0] = tracks[i][1][0] + 1
                                tracks[i][1][1] = 0

                            else:
                                tracks[i][1][0] = tracks[i][1][0] + 1
                                tracks[i][1][1] = 0

                    else:
                        if len(tracks[i][0]) > 1:
                            predictedPos.append(tracks[i][0][0] - tracks[i][0][1] + tracks[i][0][0])
                            tracks[i][0].insert(0, predictedPos[len(predictedPos)-1])
                            tracks[i][1][0] = tracks[i][1][0] + 1
                            tracks[i][1][1] = 0

                        else:
                            tracks[i][1][0] = tracks[i][1][0] + 1
                            tracks[i][1][1] = 0
                else:
                    tracks[i][1][2] = 1

        #ran out of tracklets with points left
        if len(floorPoints) > 0:
            for i in range(len(floorPoints)):
                tracks.append(([floorPoints[i]],[0, 0, 0]))

    return tracks

#tracks = objects(floorPoints, tracks)
#for i in range(len(tracks)):
#    print(tracks[i][0][0])

#print(np.array(tracks))

#l = 1