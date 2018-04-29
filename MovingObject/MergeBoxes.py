class MergeBoxes():

    def area(self,rect):
        return rect[2] * rect[3]

    def box_intersection(self,a, b):
        w = self.overlap_x(a[0], a[2], b[0], b[2])
        h = self.overlap_y(a[1], a[3], b[1], b[3])
        if w < 0 or h < 0:
            return 0
        area_box = w * h
        return area_box

    def overlap_x(self,x1, w1, x2, w2):
        w = 0
        if (x1+w1)<x2 or (x2+w2)<x1:
            return 0
        elif x2>x1 and x2+w2<x1+w1:
            w = w2
        elif (x1 + w1 - x2) >= 0:
            w = x1 + w1 - x2
        elif (w2 + x2 - x1)>=0:
            w = w2 + x2 - x1
        return w

    def overlap_y(self,y1, h1, y2, h2):
        h = 0
        if (y1+h1)<y2 or (y2+h2)<y1:
            return 0
        elif y2>y1 and y2+h2<y1+h1:
            h = h2
        elif (y1 + h1 - y2) >= 0:
            h = y1 + h1 - y2
        elif (h2 + y2 - y1)>=0:
            h = h2 + y2 - y1
        return h

    def overlap(self,x1, w1, x2, w2):
        l1 = x1 - w1 / 2.
        l2 = x2 - w2 / 2.
        left = max(l1, l2)
        r1 = x1 + w1 / 2.
        r2 = x2 + w2 / 2.
        right = min(r1, r2)
        return right - left

    def check_box_present_or_not(self,merges_box,box):
        if len(merges_box)==0:
            return True

        for b in merges_box:
            if b[0]==box[0] and b[1]==box[1]:
                return False
            else:
                return True

        return False

    def merge_boxes(self, boxes=None, max_overlap=0.6, min_overlap=0.2):
        merged_bounding_box = []

        if boxes is None:
            return None

        for i in range(len(boxes)):
            bonding_box = boxes[i]
            for j in range(len(boxes)):
                bonding_box1 = boxes[j]

                if bonding_box == bonding_box1:
                    continue

                if bonding_box1[4]==99:
                    continue

                if bonding_box[4]==99:
                    continue

                area_of_overlap = self.box_intersection(bonding_box, bonding_box1)
                a_a = self.area(bonding_box)
                b_a = self.area(bonding_box1)
                max_a = max(a_a, b_a)
                min_a = min(a_a, b_a)

                if (area_of_overlap > 0
                    and (area_of_overlap >= (min_a * max_overlap)
                         or area_of_overlap <= (max_a * min_overlap)
                         )):
                    boxes[j][4] = 99

                    if a_a > b_a:
                        max_box = bonding_box
                        if bonding_box[0]+bonding_box[2] < bonding_box1[0]+bonding_box1[2]:
                            max_box[2] = bonding_box[2] + (bonding_box1[0]+bonding_box1[2] - (bonding_box[0]+bonding_box[2]))
                        if bonding_box[1] + bonding_box[3] < bonding_box1[1] + bonding_box1[3]:
                            max_box[3] = bonding_box[3] + (bonding_box1[1] + bonding_box1[3] - (bonding_box[1] + bonding_box[3]))

                        if self.check_box_present_or_not(merged_bounding_box,max_box):
                            merged_bounding_box.append(max_box)

                    else:
                        max_box = bonding_box1
                        if bonding_box1[0] + bonding_box1[2] > bonding_box[0] + bonding_box[2]:
                            max_box[2] = bonding_box[2] + (bonding_box1[0]+bonding_box1[2] - bonding_box[0]+bonding_box[2])
                        if bonding_box1[1] + bonding_box1[3] < bonding_box[1] + bonding_box[3]:
                            max_box[3] = bonding_box1[3] + ((bonding_box[1] + bonding_box[3])-(bonding_box1[1] + bonding_box1[3]))

                        if self.check_box_present_or_not(merged_bounding_box,max_box):
                            merged_bounding_box.append(max_box)
                elif area_of_overlap > 0 and area_of_overlap >= min_a:
                    if a_a > b_a:
                        boxes[j][4] = 99
                        if self.check_box_present_or_not(merged_bounding_box, boxes[i]):
                            merged_bounding_box.append(boxes[i])
                    else:
                        boxes[i][4] = 99
                        if self.check_box_present_or_not(merged_bounding_box, boxes[j]):
                            merged_bounding_box.append(boxes[j])
                else:
                    if area_of_overlap > 0:
                        if a_a > b_a:
                            boxes[j][4] = 99
                        else:
                            boxes[i][4] = 99

        for box in boxes:
            if box[4] == 0:
                merged_bounding_box.append(box)

        return merged_bounding_box