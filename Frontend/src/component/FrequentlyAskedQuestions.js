import React from "react";
import {Collapse} from "antd";
const { Panel } = Collapse;
class FrequentlyAskedQuestions extends React.Component {
    render(){
        function callback(key) {
            console.log(key);
        }
        return(
            <div>
                <Collapse defaultActiveKey = {['1']} onChange={callback}>
                    <Panel style={{fontWeight: 'bold'}} header="How do I edit the test that I have already created?" key="2">
                      <p>If you click the "Saved test" tab, you will be redirected to a page which shows you the overall information about all of your saved tests. You can then click on "Update Test" to update a specific test.</p>
                    </Panel>
                </Collapse>
                <Collapse defaultActiveKey = {['1']} onChange={callback}>
                    <Panel style={{fontWeight: 'bold'}} header="How can I see the results of the feedback generation?" key="2">
                      <p>You will be automatically redirected to a results page whether you are generating a single feedback or a batch of feedbacks.</p>
                    </Panel>
                </Collapse>
                <Collapse defaultActiveKey = {['1']} onChange={callback}>
                    <Panel style={{fontWeight: 'bold'}} header="What is the grading system used?" key="2">
                      <p>The feedback system follows the grading mechanism thats defined by Queen Mary. This could be found under the "Grade Mechanism" tab.</p>
                    </Panel>
                </Collapse> 
                <Collapse defaultActiveKey = {['1']} onChange={callback}>
                    <Panel style={{fontWeight: 'bold'}} header="How can I change my feedback?" key="2">
                      <p>If you go under your "Saved tests" you are able to change the feedbacks which will create a new one. </p>
                    </Panel>
                </Collapse> 
                <Collapse defaultActiveKey = {['1']} onChange={callback}>
                    <Panel style={{fontWeight: 'bold'}} header="Where could I see the grade breakdown?" key="2">
                      <p>The grade breakdown is shown to you when the feedback is generated or you could view it in the "Saved feedback" page.</p>
                    </Panel>
                </Collapse> 
                <Collapse defaultActiveKey = {['1']} onChange={callback}>
                    <Panel style={{fontWeight: 'bold'}} header="How is the improvement feedback calculated?" key="2">
                      <p>The improvement feedback is entirely based on the weakest topic (s). The weakest topic is first calculated from the marks given. Then the feedback generation uses a rating system to determine how high or low the improvement feedback should be.</p>
                    </Panel>
                </Collapse>    
            </div>
        )
    }
}
export default FrequentlyAskedQuestions