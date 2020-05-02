# class Graph():

#     def addnode(self,triples):
#         for gr in triples:
#             try:
#                 self.nodes[gr['subject']] = self.nodes[gr['subject']] + [(gr['relation'], gr['object'])]
#             except:
#                 self.nodes[gr['subject']] = [(gr['relation'], gr['object'])]
#             try:
#                 self.rev_nodes[gr['object']] = self.rev_nodes[gr['object']] + [(gr['relation'], gr['subject'])]
#             except :
#                 self.rev_nodes[gr['object']] = [(gr['relation'], gr['subject'])]
#             # self.display + str(gr['subject'])

#     def query(self,triples):
#         outstr = ''
#         for i in triples:
#             # substr_no=0
#             res = [tuple(val)  for key, val in self.rev_nodes.items() if i['object'] in key or key in i['object']] 
#             try:
#                 for acceptedTriplets in set(res[0]):
#                     outstr =outstr  + acceptedTriplets[1] + ' ' + acceptedTriplets[0] + ' ' + i['object'] + '. '
#                     # substr_no = substr_no + 1
#             except:
#                 pass
#             # if len(outstr.split(' ')) > 200:
#             #     break 
#         return outstr
            

class Graph():
    def __init__(self):
        self.nodes = {}
        self.rev_nodes = {}
        self.display = ''
    def addnode(self,triples):
        for gr in triples:
            self.nodes = {}
            self.rev_nodes = {}
            self.display = ''

            try:
                self.nodes[gr['subject']] = self.nodes[gr['subject']] + [(gr['relation'], gr['object'])]
            except:
                self.nodes[gr['subject']] = [(gr['relation'], gr['object'])]
            try:
                self.rev_nodes[gr['object']] = self.rev_nodes[gr['object']] + [(gr['relation'], gr['subject'])]
            except :
                self.rev_nodes[gr['object']] = [(gr['relation'], gr['subject'])]

    def query(self,triples):
        outstr = ''
        for i in triples:
            # substr_no=0
            res = [tuple(val)  for key, val in self.rev_nodes.items() if i['object'] in key or key in i['object']] 


            try:
                for acceptedTriplets in set(res[0]):
                    outstr =outstr  + acceptedTriplets[1] + ' ' + acceptedTriplets[0] + ' ' + i['object']  + '. '
                    # substr_no = substr_no + 1
            except:
                pass
            # if len(outstr.split(' ')) > 200:
                self.rev_nodes[gr['object']] = [(gr['relation'], gr['subject'])]

    def query(self,triples):
        outstr = ''
        for i in triples:
            # substr_no=0
            res = [tuple(val)  for key, val in self.rev_nodes.items() if i['object'] in key or key in i['object']] 


            try:
                for acceptedTriplets in set(res[0]):
                    outstr =outstr  + acceptedTriplets[1] + ' ' + acceptedTriplets[0] + ' ' + i['object']  + '. '
                    # substr_no = substr_no + 1
            except:
                pass
            # if len(outstr.split(' ')) > 200:
            #     break 
        return outstr
