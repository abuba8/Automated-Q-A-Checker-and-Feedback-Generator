import React from "react"
import {Link} from "react-router-dom";
import axios from "axios";
import {Button, Table, Divider, Modal, List, Card,Collapse } from "antd";
import {CSVLink} from 'react-csv';
import '../css/Layout.css';
const { Panel } = Collapse;
class GeneratedFeedback extends React.Component {
    constructor(props)
    {
        super(props)
        this.state = {
            test: [],
            grades: [],
            generatedFeedback: [],
            answers: [],
            showingAlert: false,
            docDefinition: {},
            feedbackClicked: false,
            improvementFeedbackClicked: false,
            showOverallTable: false,
            showImprovementTable: false,
            OpenTopicPanel: false,
            topics: ""
        }
    }
    componentDidMount(){
        axios.all([
            axios.get('http://127.0.0.1:8000/api/test'),
            axios.get('http://127.0.0.1:8000/api/grades'),
            axios.get('http://127.0.0.1:8000/api/generatedFeedback'),
            axios.get('http://127.0.0.1:8000/api/answers')
        ])
        .then(axios.spread((savedtestsres, grades, feedback, answers) => {
                this.setState({
                    test: savedtestsres.data,
                    grades: grades.data,
                    generatedFeedback: feedback.data,
                    answers: answers.data
                })
                console.log(this.state.test)
                console.log(this.state.grades)
                console.log(this.state.generatedFeedback)
                console.log(this.state.answers)
        }))
    }
    saveFeedback(event1, event2) {
        console.log(event1 + "" + event2)
        
        axios.get(`http://127.0.0.1:8000/api/processnltk`)
        .then(res => {
            
            console.log(res.data)
        })
      }
    handleSave (test, grade, feedback, user, percentage) {
        let found = false
        let tempgrade = this.state.grades;
        let temptest = this.state.test;
        let grade_id = 0;
        let test_id = 0;
        //remove the percentage sign from the number
        percentage = percentage.replace(/[^\w\s]|_/g, "")
        .replace(/\s+/g, " ");
        tempgrade.map(function(gradeID, i){
            if(gradeID.grade == grade)
            {
                grade_id = gradeID.id
                temptest.map(function(testID, i){
                    if(testID.name == test)
                    {
                        test_id = testID.id
                        found = true
                        console.log("true")
                    }
                })
            }
        })
        if(found)
        {
            axios.post(`http://127.0.0.1:8000/api/save/feedback/`,{
                    test: test_id,
                    grade: grade_id,
                    user: user,
                    feedback: feedback,
                    percentage: percentage,
                    created_by: "QMFeedback",
            })
            
            let secondsToGo = 10;
            const modal = Modal.success({
              title: 'Feedback for ' + test + ' has been saved! Go check your saved feedbacks to check them out!!',
              content: `Your feedback has been saved! Go check your saved feedbacks to check them out!!`,
            });
            const timer = setInterval(() => {
              secondsToGo -= 1;
              modal.update({
                content: `This message will be destroyed after ${secondsToGo} seconds.`,
              });
            }, 1000);
            setTimeout(() => {
              clearInterval(timer);
              modal.destroy();
            }, secondsToGo * 1000);
        }
        else
        {
            alert("Could not save the feedback!")
        }
        console.log(test + " " + grade + " " + feedback + " " + user + " " + percentage)
    }

    handleSaveImprovement(test, grade, area, feedback, user) {
      let found = false
      let tempgrade = this.state.grades;
      let temptest = this.state.test;
      let grade_id = 0;
      let test_id = 0;
      tempgrade.map(function(gradeID, i){
          if(gradeID.grade == grade)
          {
              grade_id = gradeID.id
              temptest.map(function(testID, i){
                  if(testID.name == test)
                  {
                      test_id = testID.id
                      found = true
                      console.log("true")
                  }
              })
          }
      })
      if(found)
      {
          axios.post(`http://127.0.0.1:8000/api/save/improvement/`,{
                  test: test_id,
                  area_of_improvement: area,
                  user: user,
                  improvement_feedback: feedback,
          })
          
          let secondsToGo = 10;
          const modal = Modal.success({
            title: 'The improvement Feedback for ' + test + ' has been saved! Go check your saved feedbacks to check them out!!',
            content: `Your feedback has been saved! Go check your saved feedbacks to check them out!!`,
          });
          const timer = setInterval(() => {
            secondsToGo -= 1;
            modal.update({
              content: `This message will be destroyed after ${secondsToGo} seconds.`,
            });
          }, 1000);
          setTimeout(() => {
            clearInterval(timer);
            modal.destroy();
          }, secondsToGo * 1000);
      }
      else
      {
          alert("Could not save the improvement feedback!")
      }
      console.log(test + " " + grade + " " + feedback + " " + user )
  }
  handleFeedbackPopUp(e, val){
    if(val == "overall")
    {
      this.setState({feedbackClicked: true})
    }
    else
    {
      this.setState({improvementFeedbackClicked: true})
    }
  }
  handleTopics(testName){
    let test_id = 0;
    let list_of_topics = []
    axios.get(`http://127.0.0.1:8000/api/test`)
        .then(res => {
            res.data.map(function(item, i){
              if(testName == item.name)
              {
                test_id = item.id
              }
            })
            axios.get(`http://127.0.0.1:8000/api/answers`)
            .then(res => {
                res.data.map(function(item, i){
                  if(item.test == test_id)
                  {
                    list_of_topics = item.topics
                  }
                })
                this.setState({
                  topics: list_of_topics,
                  OpenTopicPanel: true
                })
            })
        })
  }
  handleOverallClick(){
    this.setState({
      feedbackClicked: false,
    })
  }
  handleImprovementClick(){
    this.setState({
      improvementFeedbackClicked: false
    })
  }
    render(){
      function callback(key) {
        console.log(key);
      }
        const {id} = this.props.match.params
        console.log(id)
        var list_topics = new Array();
        list_topics = this.state.topics.split(",")
        const data = [];
        list_topics.map((i,j) =>{
          data.push({title: list_topics[j]})
        })
        const columns = [{
            title: 'Test Name',
            dataIndex: 'test',
            key: 'test',
            render: (text) => {
              return {
                children: text,
                props: {
                  'data-tip': 'a very long text',
                },
              };
            },
          }, {
            title: 'Grade given',
            dataIndex: 'grade',
            key: 'grade',
          },
          {
            title: 'Feedback percentage',
            dataIndex: 'percentage',
            render: (text) => {
              return {
                children: text,
                props: {
                  'data-tip': 'a very long text',
                },
              };
            },
          }, {
            title: 'Feedback Generated',
            dataIndex: 'feedback',
            render: (text, record) => (
                  <span>
                    <Button onClick={(e) => this.handleFeedbackPopUp(this.props.match.params.feedback, "overall")} htmlType="submit">View Overall feedback</Button>
                  </span>
            ),
          }, {
            title: 'Action',
            key: 'action',
            render: (text, record) => (
              <span>
                <Button onClick={(e) => this.handleSave(record.test, record.grade, this.props.match.params.feedback, this.props.match.params.userid, record.percentage)} type="primary" htmlType="submit">Save Feedback</Button>
                <Divider type="vertical" />
                <Link to={`/createFeedback/` + this.props.match.params.testid + '/' + this.props.match.params.userid}><Button type="primary" htmlType="submit" style={{alignItems:'center'}}>Not happy with this feedback?</Button></Link>
                <Divider type="vertical" />
                <CSVLink data={this.props.match.params.feedback} ><Button type="primary" htmlType="submit" >Export Overall Feedback</Button></CSVLink>
              </span>
            ),
          }];
          const Imrpovementcolumns = [ {
            title: 'Area (s) of improvement',
            dataIndex: 'areaOfImprovement',
            key: 'areaOfImprovement',
          },{
            title: 'Imrpovement Feedback',
            dataIndex: 'feedback',
            key: 'feedback',
            render: (text) => (
              <span>
                <Button onClick={(e) => this.handleFeedbackPopUp(this.props.match.params.feedback, "improvement")} htmlType="submit">View Improvement feedback</Button>     
              </span>
            ),
          },{
            title: 'Other topics',
            dataIndex: 'topics',
            key: 'topics',
            render: (text) => (
              <span>
                {
                  this.state.OpenTopicPanel ?
                  <List
                            grid={{ gutter: 16, column: 2 }}
                            dataSource={data}
                            renderItem={item => (
                              <List.Item>
                                  Title: <h8 style={{color: '#096dd9'}}>{item.title}</h8>
                              </List.Item>
                            )}
                        />
                  :
                  <span>
                  <Button onClick={(e) => this.handleTopics(this.props.match.params.testid)} htmlType="submit">View other topics</Button>     
                </span>
                }
              </span>
            ),
          },{
            title: 'Action',
            key: 'action',
            render: (text, record) => (
              <span>
                <Button onClick={(e) => this.handleSaveImprovement(this.props.match.params.testid, this.props.match.params.testgrade, record.areaOfImprovement, this.props.match.params.improvement, this.props.match.params.userid)} type="primary" htmlType="submit">Save Improvement Feedback</Button>
                <Divider type="vertical" />
                <Link to={`/createFeedback/` + this.props.match.params.testid + '/' + this.props.match.params.userid}><Button type="primary" htmlType="submit" style={{alignItems:'center'}}>Not happy with this improvement feedback</Button></Link>
                <Divider type="vertical" />
                <CSVLink data={this.props.match.params.improvement} ><Button type="primary" htmlType="submit" >Export Improvement Feedback</Button></CSVLink>
              </span>
            ),
          }];
          let score = 0;
          
          if(this.props.match.params.score.toString().length == 2)
          {
            score = this.props.match.params.score;
          }
          else if(this.props.match.params.score < 0)
          {
            if(this.props.match.params.score.toString() > 2)
            {
              score = Math.round(this.props.match.params.score)
            }
            score = this.props.match.params.score;
          }
          else
          {
            
            score = Math.round(this.props.match.params.score * 100)
          }
          console.log(this.props.match.params.testid + this.props.match.params.feedback + this.props.match.params.improvement)
          const testInfo = [{
            key: '1',
            test: this.props.match.params.testid,
            grade: this.props.match.params.testgrade,
            percentage: score + "%"
          }];
          const improvementInfo = [{
            key: '1',
            areaOfImprovement: this.props.match.params.topicImprovement
          }];
        return(
            <div>
                <h2 style={{ display: 'flex', justifyContent: 'center'}} >Here's a summary of your chosen feedback</h2>
                <Link to={`/reviewFeedback/` + this.props.match.params.testid + `/` + this.props.match.params.testmark +`/` + this.props.match.params.testgrade + `/` + this.props.match.params.correct +`/`+ this.props.match.params.incorrect +`/` + this.props.match.params.effect  + `/` +this.props.match.params.userid}><img width="30" height="30" src="https://img.icons8.com/flat_round/64/000000/back--v1.png"/></Link>
        
                <Table id="test" columns={columns} dataSource={testInfo} />
                {
                this.state.feedbackClicked ?
                <div>
                  <Collapse defaultActiveKey = {['1']} onChange={callback}>
                    <Panel header="Overall Feedback" key="2">
                      <p>{this.props.match.params.feedback}</p>
                    </Panel>
                    
                  </Collapse><br/>
                  <div style={{marginLeft: '40%', marginRight: '50%'}}>
                    <Button onClick={(e) => this.handleOverallClick()} type="primary">Close feedback</Button>
                  </div>
                </div>
                :
                null
              }
              {/*   ############IMPROVEMENT TABLE############      */}
              <h2 style={{display: 'flex', justifyContent: 'center'}} >Here's a summary of your chosen improvement feedback</h2>
                <Table id="test2" columns={Imrpovementcolumns} dataSource={improvementInfo} />
              {
                this.state.improvementFeedbackClicked ?
                <div>
                    <Collapse defaultActiveKey = {['1']} onChange={callback}>
                      <Panel header="Improvement Feedback" key="2">
                        <p>{this.props.match.params.improvement}</p>
                      </Panel>
                      
                    </Collapse><br/>
                    <div style={{marginLeft: '40%', marginRight: '50%'}}>
                      <Button onClick={(e) => this.handleImprovementClick()} type="primary">Close feedback</Button>
                    </div>
                  </div>
                  :
                  null
              }
              
            </div>
        )
    }
}
export default GeneratedFeedback
