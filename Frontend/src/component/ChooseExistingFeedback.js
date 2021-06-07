import React from "react";
import { Form, Button,Card, Col, Row, Spin, Typography, Modal, List, Input  } from "antd";
import axios from "axios";
import {Link} from "react-router-dom";
import '../css/Layout.css';
import '../css/reviewFeedback.css';
import { LoadingOutlined } from '@ant-design/icons';
import FeedbackInput from "./GeneralInput.js"
const antIcon = <LoadingOutlined style={{ fontSize: 24 }} spin />;
const {Text} = Typography;

class ChooseExistingFeedback extends React.Component{
    constructor(props){
        super(props)
        this.state = {
            data: [],
            showFeedback: false,
            improvement: false,
            effectiveness: "",
            feedbackAmount: 0,
            feedbackAmountState: false,
            collectionFeedback: true,
            value: '',
            improvementFeedback: [],
            AreasOfImprovement: "",
            marks: "",
            topics: "",
            giveImprovement: false,
            personalImprovementGiven: false,
            personalImprovement: "",
        }
    }
    onChange = value => {
      this.setState({ value });
    };
    componentDidMount(){
      
    }
    generateFeedback(testName, testGrade, testMark, correct){
        console.log("hi yes!!")
        
        this.setState({
            showFeedback: false,
        })
        axios.get(`http://127.0.0.1:8000/api/processnltk/${testName}/${testGrade}/${testMark}/${correct}`)
        .then(res => {
            console.log(res.data)
            var random_feedback = res.data[Math.floor(Math.random()*res.data.length)];
            if(random_feedback.category == "negative")
            {
              this.setState({
                improvement: true,
              })
            }
            this.setState({
                data: random_feedback,
                showFeedback: true
            })
        })
    }
    handleFormSubmit = (e, testName, testGrade, testMark, correct,incorrect, effect) => {
        e.preventDefault();
        this.setState({
          collectionFeedback: true,
        })
        const total = this.state.value;
        let random_data = [];
        let temp_feedback = [];
        let improvement_feedback = [];
        console.log(total);
        let test_id = 0;
        let list_of_marks = []
        let list_of_topics = []
        let weakest = ""
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
                    list_of_marks = item.topic_mark_breakdown
                    weakest = item.weakest_topic
                  }
                })
                this.setState({
                  AreasOfImprovement: weakest,
                  topics: list_of_topics,
                  marks: list_of_marks
                })
            })
        })
        axios.get(`http://127.0.0.1:8000/api/processnltk/${testName}/${testGrade}/${testMark}/${correct}/${incorrect}/${effect}`)
        .then(res => {
            console.log(res.data)
            temp_feedback = res.data[0]
            improvement_feedback = res.data[1]
            var improvement = improvement_feedback[Math.floor(Math.random()*improvement_feedback.length)]
            for(let i = 0; i < total; i++)
            {
                var random_feedback = temp_feedback[Math.floor(Math.random()*temp_feedback.length)];
                random_data.push(random_feedback)
            }
            this.setState({
                data: random_data,
                collectionFeedback: false,
                improvementFeedback: improvement,
                giveImprovement: true,
            })
        })
        this.setState({
            feedbackAmountState: true,
            feedbackAmount: total
        })
    }
    onKeyPress(event) {
        const keyCode = event.keyCode || event.which;
        const keyValue = String.fromCharCode(keyCode);
         if (/\+|-/.test(keyValue))
           event.preventDefault();
      }
    handleImprovementSubmit(e, requestMethod, test, mod)
    {
        e.preventDefault();
        const title = e.target.elements.name.value;
        console.log(title);
        this.setState({
          personalImprovementGiven: true,
          personalImprovement: title
        })
    }
      createImprovement(test){
        let secondsToGo = 600;
        
        const modal = Modal.success({
          title: `This pop up will be destroyed after 10 minutes` ,
          content: '',
        });
        const timer = setInterval(() => {
          secondsToGo -= 1;
          modal.update({
            content: 
            <div>
            <div className={`alert alert-success ${this.state.showingAlert ? 'alert-shown': 'alert-hidden'}`}>
                  <strong>Your improvement feedback has been changed!</strong>
            </div>
            <h2 style={{display: 'flex', justifyContent: 'center'}} >Create a improvement feedback for {test} here!</h2>
            <Form onSubmit={(event) => this.handleImprovementSubmit(event, test)}>
                <Form.Item label="Improvement Feedback">
                  <Input name="name" placeholder="write your feedback here.." />
                </Form.Item>
                <Form.Item>
                  <Button type="primary" htmlType="submit">Create improvement feedback</Button>
                </Form.Item>
          </Form>
        </div> ,
          });
        }, 1000);
        setTimeout(() => {
          clearInterval(timer);
          modal.destroy();
        }, secondsToGo * 1000);
      }
    
    render()
    {
      let testName = this.props.match.params.testid
      let testGrade = this.props.match.params.testgrade
      let testMark = this.props.match.params.testmark
      let correct = this.props.match.params.correct
      let incorrect = this.props.match.params.incorrect
      let score = this.props.match.params.score;
      let effect = this.props.match.params.effect;
      let user = this.props.match.params.userid;
      var list_topics = new Array();
      var list_marks = new Array();
      list_topics = this.state.topics.split(",")
      list_marks = this.state.marks.split(",")
      let Finalimprovement = this.state.improvementFeedback.review;
      let Area = this.state.AreasOfImprovement;
      const data = [];
      list_topics.map((i,j) =>{
        data.push({title: list_topics[j], mark: list_marks[j]})
      })
        return(
            <div >
                
                {
                this.state.feedbackAmountState ? 
                    
                <Row gutter={10} justify="space-around" type="flex">
                  <Col span={5}>
                    <Card className = "popupreview" bordered style={{color: 'blue'}} title="Test Information" bordered={true}>
                      <Text strong>Test Name:</Text> <Text strong style={{color: '#096dd9'}}>{this.props.match.params.testid}</Text> <br/>
                      <Text strong>Test Grade:</Text> <Text strong style={{color: '#096dd9'}}>{this.props.match.params.testgrade}</Text> <br />
                      <Text strong>Test Mark:</Text> <Text strong style={{color: '#096dd9'}}>{this.props.match.params.testmark} %</Text> <br/>
                      <Text strong>Correct Answers:</Text> <Text strong style={{color: '#096dd9'}}>{this.props.match.params.correct}</Text> <br/>
                      <Text strong>Incorrect Answers:</Text> <Text strong style={{color: '#096dd9'}}>{this.props.match.params.incorrect}</Text> <br/>
                    </Card>
                  </Col>
                    {/* Test breakdown */}
                  <Col>
                    <Card className="popupreview" bordered style={{color: 'blue', textAlign: 'center'}} title="Test breakdown" bordered={false}>
                        <List
                            grid={{ gutter: 16, column: 2 }}
                            dataSource={data}
                            renderItem={item => (
                              <List.Item>
                                <Card title={item.title}>Mark: <h8 style={{color: '#096dd9'}}>{item.mark}</h8></Card>
                              </List.Item>
                            )}
                        />
                    </Card>
                  </Col>
                  {/* Improvement Information */}
                  
                  <Col>
                    {
                    this.state.personalImprovementGiven ? 
                    <Card className="popupreview" bordered style={{color: 'blue',marginLeft: 10}} title="Improvement Information" bordered={false}>
                        <Text strong>Area (s) of improvement: </Text>
                        <Text strong style={{color: '#096dd9'}}>{this.state.AreasOfImprovement}</Text><br />
                        <Text strong>Created by: </Text>
                        <Text strong style={{color: '#096dd9'}}>{this.props.match.params.userid}</Text>
                    </Card>
                    :
                    <Card className="popupreview" bordered style={{color: 'blue',marginLeft: 10}} title="Improvement Information" bordered={false}>
                      <Text strong>Area (s) of improvement:</Text>{this.state.giveImprovement ? <Text strong style={{color: '#096dd9'}}>{this.state.AreasOfImprovement}</Text> : <Spin indicator={antIcon} />} <br/>
                      <Text strong>Improvement:</Text> {this.state.giveImprovement ? <Text strong style={{color: '#096dd9'}}>{this.state.improvementFeedback.level}</Text> : <Spin indicator={antIcon} />} <br/>
                      <Text strong>Outcome of the generator:</Text> {this.state.giveImprovement ? <Text strong style={{color: '#096dd9'}}>{this.state.improvementFeedback.category}</Text>: <Spin indicator={antIcon} />} <br />
                    </Card> 
                    }
                       
                  </Col> 
                  {/* Information Feedback*/}
                  <Col span={8}>
                    <Card className="popupreview" headStyle={{backgroundColor: 'red'}} bordered style={{color: 'blue'}} title="Improvement Feedback" bordered={false}>
                        
                        <Text strong>Feedback: 
                          {
                          this.state.giveImprovement 
                          ? 
                          <Text strong style={{color: '#096dd9'}}>
                            {this.state.improvementFeedback.review}
                          </Text> 
                          : 
                          <Spin indicator={antIcon} />
                          }
                        </Text>
                    </Card>
                  </Col>
                  
                  {
                    this.state.collectionFeedback ?
                   <Text strong style={{color: 'skyblue', fontSize: '19px'}}>Please wait while {this.state.feedbackAmount} feedback(s) are being generated!<br/><Spin indicator={antIcon} /></Text>
                    :
                    <div style={{textAlign: 'center'}}>
                      {this.state.data.map(function(item, i){
                        let title = "Feedback " + (i+1)
                        let score = Math.round(item.score * 100)
                        return(
                          <Col span={5} style={{textAlign: 'center'}}>
                            <Card bordered style={{ color: 'blue'}} title={title} bordered={false}>
                                <Text strong style={{color: '#096dd9'}}>Review: </Text><Text strong>{item.review}</Text><br />
                                <Text strong style={{color: '#096dd9'}}>Score: </Text><Text strong>{score}</Text><br />
                                <Text strong style={{color: '#096dd9'}}>Outcome of test: </Text><Text strong>{item.effectiveness}</Text>
                            </Card>
                            <Link to={`/generatefeedback/` + testName + `/` + testMark +`/` + testGrade + `/` + correct +`/`+ incorrect +`/` + score + `/` + item.review + `/` + Finalimprovement + `/` + Area + `/` + effect + `/` + user}><Button style={{marginLeft: '70%', margin: '5px'}} type="primary">Happy to see the full result?</Button></Link>
                          </Col>
                        )
                      })}
                      
                    </div>
                  }
                </Row>
                 : 
                 <Form onSubmit={(event) => this.handleFormSubmit(event, this.props.match.params.testid, this.props.match.params.testgrade, this.props.match.params.testmark, this.props.match.params.correct, this.props.match.params.incorrect, this.props.match.params.effect)}>
                    <Form.Item style={{textAlign: 'center'}} label="How many feedbacks would you like to see? (maximum 5)">
                        <FeedbackInput name="amount" style={{ width: 120 }} value={this.state.value} onChange={this.onChange} />
                    </Form.Item>
                    <Form.Item style={{textAlign: 'center'}} >
                        <Button type="primary" htmlType="submit">Generate Feedbacks</Button>
                    </Form.Item>
                 </Form>
            
                }
            </div>
        )
    }
}
export default ChooseExistingFeedback