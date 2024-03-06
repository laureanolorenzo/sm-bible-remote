function prettifyPassage(passage){
    let passageChunks = passage.split(`/\[\d+:\d+\]/`);
    let resultPassage = []
    for (let passage of passageChunks) {
        resultPassage.push(`<p class="passage">${passage}</p><br>`)
    }
    return resultPassage.join('\n')
}
function createHTMLFromResponse(results){
    let htmlResponse = `        
    <div class="agent-interaction">
        <div class="agent-profile">
            <img src="static/images/openArtPriest.png" alt="Chatbot's profile picture" class = "agent-pic">
            <p><b>AI</b></p>
        </div>`;
    
    let status = JSON.parse(results.status);
    if (status == 200){
        let docs = results.relevant_documents;
        docs = Array.from(docs);
        if (docs.length) {
            htmlResponse += `
            <div class="agent-message">
                <h3>I've found some verses that could match what you're looking for:</p>  
            `
            for (let i = 0; i < docs.length; i++) {
                htmlResponse += `
                    <h4>Passage number: ${i+1}</h4>
                    <h4><i>${docs[i].book}, ${docs[i].chapter}</i></h4>
                    ${prettifyPassage(docs[i].passage)}
                    `            
            }
        }
    } else if (status == 201) {
        htmlResponse += `
        <div class="agent-message">
            <p>I'm sorry, it seems I couldn't find relevant verses. Please try with a different question.</p>  
        `
    } else if (status == 401) {
        htmlResponse += `
        <div class="agent-message">
            <p>I'm sorry, it seems the entered authorization key is not valid.</p>  
        `
    } else {
     htmlResponse += `
            <div class="agent-message">
                <p>I'm sorry, it seems the server is not responding</p>  
            </div>
        `
    }
    htmlResponse += `
    </div>
    `
    return htmlResponse;
    

}
async function displayNewUserMessage(message,chatHistory,userInputBar,authInput,tInput,nInput) {
    displayingMessage = true; //Disable chat while waiting for api result
    
    if (message.value.length) {
        await new Promise(resolve => setTimeout(resolve,50));
        chatHistory.innerHTML += `
        <div class="user-interaction">
            <div class="user-message">
                <p>${message.value}</p>
            </div>
        </div> 
        `
        let messageForAPI = message.value;
        userInputBar.value = 'Type a message here'; //Reset message
        chatHistory.innerHTML += `
        <div class="agent-interaction" id="loading-child">
            <div class="agent-profile">
                <img src="static/images/openArtPriest.png" alt="Chatbot's profile picture" class = "agent-pic">
                <p><b>AI</b></p>
            </div>
            <div class="agent-message">
                <div class = "loader"></div>
            </div>
        </div>
        ` 
        let results = await fetch(`${window.location.href}/predict`, {
            method: 'POST',
            headers: {
                "Content-Type": "application/json",
                "Auth": authInput
              },
            body: JSON.stringify({message:messageForAPI,n_results:nInput,threshold:tInput})
        });
        results = await results.json()
        let loadingElem = document.getElementById('loading-child');
        await chatHistory.removeChild(loadingElem);
        chatHistory.innerHTML += createHTMLFromResponse(results);
        displayingMessage = false;
    }
}
let displayingMessage = false; //Outside window event listener, otherwise it gets redifined as false!
window.addEventListener('load', () => {
    const userInputBar = document.querySelector('.user-input-bar');
    const sendBtn = document.querySelector('.send-btn');
    let authBtn = document.querySelector('.auth-btn');
    let authInput;
    authBtn.addEventListener('click', ()=> {
        authInput = document.querySelector('.auth').value;
    })
    let tBtn = document.querySelector('.threshold-btn');
    let tInput;
    authBtn.addEventListener('click', ()=> {
        tInput = document.querySelector('.threshold').value;
    })
    let nBtn = document.querySelector('.n-results-btn');
    let nInput;
    nBtn.addEventListener('click', ()=> {
        nInput = document.querySelector('.n-results').value;
    })
    // authObjInput.addEventListener('input keyup change', () =>
    // {authInput = authInputInput.value})
    let chatHistorySection = document.querySelector('.chat-history-section');
        userInputBar.addEventListener('focus', ()=> {
            if (userInputBar.value == 'Type a message here') {
                userInputBar.value = '';
            }
        });
        userInputBar.addEventListener('blur', ()=> {
            if (userInputBar.value == '') {
                userInputBar.value = 'Type a message here';
            }
        });
        sendBtn.addEventListener('click',()=>
        {   
            if ((userInputBar.value != 'Type a message here') && (!displayingMessage)){
                displayNewUserMessage(userInputBar,chatHistorySection,userInputBar,authInput,tInput,nInput);
            }
        });
        window.addEventListener('keydown',(e)=>
        {
            if ((e.key == 'Enter') && (userInputBar == document.activeElement) && (!displayingMessage)) {
                displayNewUserMessage(userInputBar,chatHistorySection,userInputBar,authInput,tInput,nInput);
            } else if ((userInputBar == document.activeElement) && (userInputBar.value == 'Type a message here') && (e.key != 'Enter')) {
                userInputBar.value = '';
            }
        });
});