import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from pinecone_utils import *
## Embeddings Models
def filter_stopwords(chunk,lang = 'english'):
    try: 
        stop_words = set(stopwords.words(lang))
    except: #May need to install it if it's the first time you use it
        nltk.download('stopwords')
        nltk.download('punkt')
        stop_words = set(stopwords.words(lang))
    chunks_only = True #Wether to create a text completion based on the relevant chunks
    metric_threshold = -6 # If in any given interaction no chunk  reaches this degree of similarity to the question, then
    filtered_words = []
    word_tokens = word_tokenize(chunk)
    for word in word_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    return ' '.join(filtered_words)
n_results_passed = 3 ## Max passages returned 
def get_context(query_results,n_results_passed = 3,cross_scores = True):
    context = []
    for n, (text, cross_score,distance,book,chapter) in enumerate(zip(query_results['passages'][:n_results_passed],query_results['cross_scores'][:n_results_passed],query_results['distances'][:n_results_passed],query_results['books'][:n_results_passed],query_results['chapters'][:n_results_passed])): 
        indiv_context = {}
        text = ''.join([s for s in text.splitlines(True) if s.strip('\r\n')])
        indiv_context['passage'] = text
        indiv_context['distance'] = np.float64(distance)
        if cross_scores:
            indiv_context['cross_scores'] = np.float64(cross_score)
        indiv_context['book'] = book
        indiv_context['chapter'] = chapter
        context.append(indiv_context)
    #For now we don't worry about other relevant texts
    return context
def semantic_search(query_embeddings,index,re_rank = True, threshold = None,prompt = None,n=50,top_n = 3,lang='english',doc_name = 'sb_test'): #Should access a global variable "collection"
    if n > 50:
        n = 50 #50 is enough!
    query_results = index.query( #See https://docs.pinecone.io/reference/query
        namespace= doc_name,
        top_k=n,
        include_metadata=True,
        vector=[q for q in query_embeddings],
    )
    
    query_results = query_results['matches']
    mapped_results = {
        'passages':list(map(lambda x:x['metadata']['passage'],query_results)),
        'distances': list(map(lambda x:x['score'],query_results)),
        'books': list(map(lambda x:x['metadata']['book'],query_results)),
        'chapters': list(map(lambda x:x['metadata']['chapter'],query_results))
    }
    return mapped_results #So far decent results. Ada model is still better though.
def get_second_results(sorted_passages:list,threshold,mapped_results,top_n = 10): #Handle models outside function!
    if top_n > len(sorted_passages):
        top_n = len(sorted_passages)
    sorted_passages = dict(sorted_passages)
    mapped_results['cross_scores'] = [sorted_passages[passage] for passage in mapped_results['passages']]

    sorted_items = sorted(zip(mapped_results['passages'],mapped_results['cross_scores'],mapped_results['distances'],mapped_results['books'],mapped_results['chapters']),key=lambda x:x[1],reverse=True) #Sort by second score (cross score)
    
    new_results = {
                'passages':[item[0] for item in sorted_items[:top_n] if item[1] > threshold],
                'cross_scores':[item[1] for item in sorted_items[:top_n] if item[1] > threshold],
                'distances':[item[2] for item in sorted_items[:top_n] if item[1] > threshold],
                'books': [item[3] for item in sorted_items[:top_n] if item[1] > threshold],
                'chapters':[item[4] for item in sorted_items[:top_n] if item[1] > threshold]
                }
    return new_results
if __name__ == '__main__':
    vectors = [-0.0434880517,0.101962395,-0.0275958758,-0.0309239179,-0.0258174594,-0.0220498182,0.021885708,0.00184764282,-0.0153137157,0.0425264575,-0.0311417,-0.0192874968,0.032358259,-0.0856483206,-0.0880338475,-0.0742117241,-0.0308561455,0.0632938743,-0.00293572736,-0.108697854,-0.0157327075,0.0616940558,0.0221736953,-0.0462682098,0.114249691,0.0272796955,-0.0203846842,-0.0832972154,0.10214626,-0.0120990546,-0.0637891442,0.0188643057,0.00663703401,-0.0324289426,-0.0526957624,0.135394931,0.0120645938,-0.00344659388,-0.0103962263,-0.00430295756,0.0143574374,0.0438187122,0.026621256,0.00176566676,-0.0434991233,0.0320828557,-0.0327964574,-0.0133367162,-0.0232931431,0.0304150525,-0.110931292,0.0425142087,0.0226285141,-0.00632541394,-0.0211159289,0.0456111,0.000719347,-0.0160421804,0.0595438741,-0.0306060538,-0.0887841284,0.043042805,-0.0185938179,-0.00989978109,0.0852897316,0.006034825,0.0330357552,-0.015663974,-0.112465918,0.0461736694,0.0433176681,0.0661580637,0.0574667789,-0.0707618371,-0.110448666,-0.07929454,0.0322702304,-0.0537243672,-0.0318589061,0.00119683752,0.0519848652,0.0369499922,0.0494639575,0.0484195352,-0.0414771102,-0.0466968119,0.0295497812,0.0217501279,0.101599254,-0.00797532871,0.0554048903,-0.0588055402,0.019960219,0.0223326497,-0.0213668365,0.00225514732,0.00142211991,0.0799656659,-0.00186931039,0.0381067134,-0.00766467676,0.0104249101,-0.0294160116,0.0237826388,0.00275251083,0.0105945254,-0.0432948694,0.053421028,-0.0146298343,-0.0191928614,0.0466931351,-0.0845955312,-0.0221587438,-0.00789745,0.0220935121,0.0752796754,0.0750498101,-0.0200710185,-0.0498495288,0.0249456093,0.0495294631,0.0694702268,0.0172751788,0.00326920371,-0.0459521376,0.0508234203,0.0311635714,1.77551674e-31,0.0748677254,0.0000399687488,-0.0447906032,-0.0375866145,0.069339864,0.0526698045,-0.024329327,0.0649366826,-0.101450138,-0.0566991381,0.0619293228,-0.0809564069,-0.0453993864,-0.016935261,-0.0026505522,-0.0226522312,0.0488691442,-0.0561925285,0.0891438648,-0.0626755208,-0.00401091063,0.0358607285,-0.0803992823,-0.134820655,-0.0234486721,-0.044431068,0.0898197666,-0.0184099674,-0.0564433374,-0.0237178616,0.00420409767,-0.0264696907,0.00771218538,0.036334984,0.0798910931,-0.010952523,-0.0441818126,0.00589717,0.0328680128,-0.0185606014,-0.0290591,0.00889722444,0.013215025,-0.0740639418,-0.034822844,0.00120548985,-0.021731453,-0.0201236,-0.00504165282,0.0526206382,-0.0201577358,0.0241449159,0.0576782636,-0.0843182281,0.0316296034,0.0642125905,-0.00653355569,0.0991636589,0.021123074,0.0464574546,-0.056677606,-0.0210978463,0.0398850702,0.0962097645,-0.0140132699,-0.0759087,0.00431345031,-0.0233054888,0.0431399047,0.0336731337,-0.0751160905,0.0176759176,-0.0200246759,0.0203324016,0.0671609491,-0.0249734297,-0.0288274493,-0.0801624283,-0.0778045207,0.0158223528,-0.0108601497,0.028860582,-0.015450662,0.0561999455,0.0997647,0.0298231319,-0.0237062294,0.0221304372,0.0524718873,-0.0597140044,0.0452270955,-0.0469363369,0.0958313793,-0.117191747,-0.050815478,-3.41815427e-33,0.0835716352,-0.0295658223,0.00190529134,0.00103746913,-0.0257710814,-0.158648312,-0.0131604271,0.113909163,-0.0200310107,0.00813987572,-0.0270989984,0.0308681838,-0.0144757489,-0.0518249385,-0.044831872,-0.0316775218,0.0241609402,0.00719610648,0.0732554272,0.0115356725,0.0127537018,-0.0340913795,0.0284185745,0.0690466166,0.10507632,-0.0898512378,0.0509660691,0.0612494312,-0.046414759,-0.0108048087,0.017286174,-0.0683023185,0.00947937556,0.0701196641,-0.0574288964,-0.033045657,0.0494305938,0.0411079153,0.0105719753,-0.0175539106,-0.00635293964,0.0456535779,0.0454281569,-0.0496936888,-0.0756791905,0.0000278130319,0.0460397601,0.0867088661,-0.0233896971,-0.0202927608,-0.0628056,-0.0204611309,0.0302541126,0.0625694171,0.00207458087,-0.0636145771,0.0458199531,-0.0298563652,0.00414810795,0.00983720645,0.0260270517,-0.0425468646,-0.024315061,0.043395251,0.136923671,0.0367174856,-0.063211605,0.0108274147,-0.0412295945,0.0497017317,-0.0447423831,-0.161569506,-0.0349276625,0.0208982024,0.0141118513,0.00161189807,0.0450139977,-0.00792324,0.0550445691,-0.0288799163,0.107392572,-0.025629716,-0.0545275621,-0.0332863927,-0.00488571776,-0.0891367346,0.0601044595,-0.022635024,-0.0495235324,-0.00483080791,-0.0232160129,0.0126531916,-0.0513002574,0.0331747159,0.0120603461,-2.64219355e-33,-0.00737353973,0.0093703568,-0.0668314248,-0.0039279405,0.112069175,-0.0741210207,0.0846013576,-0.0289540291,-0.0154346,-0.00427445024,-0.00121082657,-0.00357982144,0.0456625223,-0.0260893237,0.0683166832,-0.0502910167,-0.0348937,-0.090014495,-0.0379603915,-0.0281957835,0.0468303971,-0.0712249726,0.0576271452,0.0171570424,-0.0963151678,0.0426450968,0.0117485262,-0.0423598811,0.054550536,0.0405069888,0.0220709946,0.049428463,-0.0659608394,0.0103585143,-0.048445642,-0.00214782567,-0.0895040855,-0.0571135916,0.0204566475,-0.0361446589,0.0146980174,0.0499045737,0.0195138529,0.00828291662,0.00494442228,-0.0715058818,0.0196935087,0.097479023,0.010429726,0.0491952151,0.0212471541,0.105399512,0.0897078,-0.0367468819,-0.0242451336,-0.0430022292,0.0960122645,0.0196152031,-0.0643016249,-0.0405381024,0.0245694015,0.00358547922,0.0471415482,-0.0657792091]
    result = semantic_search(vectors,pc_index,)
    print(result)
    print('LEN',len(result['passages']))
    sorted_passages = []
    for n,p in enumerate(result['passages']):
        sorted_passages.append((p,n+1))
    second_res = get_second_results(sorted_passages,2,result)
    print(second_res)
    print(len(second_res['passages']))









# def get_relevant_docs(text,completions = False):
#     query_results = semantic_search(text,pc_index,re_rank= True,threshold = metric_threshold,top_n=3,doc_name='sb_test')
#     if not len(query_results):
#         return False
#     if not completions:
#         return get_context(query_results,0,cross_scores=True)
