import React from "react";
import { Statistic, Form, Button, Input,Card, Col, Row, Spin, Typography, Modal, Tooltip, List  } from "antd";
import axios from "axios";
import {Link} from "react-router-dom";
import '../css/Review.css';
import { LoadingOutlined } from '@ant-design/icons';
const antIcon = <LoadingOutlined style={{ fontSize: 24 }} spin />;
const {Text} = Typography;
class ReviewFeedback extends React.Component{
    constructor(props){
        super(props)
        this.state = {
            feedbackData: [],
            improvementFeedbackData: [],
            showFeedback: false,
            improvement: false,
            giveImprovement: false,
            effectiveness: "",
            AreasOfImprovement: "",
            personalImprovementGiven: false,
            personalImprovement: "",
            editedFeedback: "",
            editedFeedbackGiven: false,
            typeOfFeedback: false,
            feedbackColor: "",
            topics: "",
            marks: "",
            question_type: "",
        }
        this.onChangeFeedback = this.onChangeFeedback.bind(this)
    }
    componentDidMount(){
        const testName = this.props.match.params.testid
        const testGrade = this.props.match.params.testgrade
        const testMark = this.props.match.params.testmark
        const correct = this.props.match.params.correct
        const incorrect = this.props.match.params.incorrect
        const effectiveness = this.props.match.params.effect
        let temp_feedback = [];
        let improvement_feedback = [];
        let test_id = 0
        let weakest = ""
        let list_of_topics = []
        let list_of_marks = []
        let questiontype = "";
        axios.get(`http://127.0.0.1:8000/api/test`)
        .then(res => {
            res.data.map(function(item, i){
              if(testName == item.name)
              {
                test_id = item.id
                questiontype = item.questiontype
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
                  marks: list_of_marks,
                  question_type: questiontype
                })
            })
        })
        axios.get(`http://127.0.0.1:8000/api/processnltk/${testName}/${testGrade}/${testMark}/${correct}/${incorrect}/${effectiveness}`)
        .then(res => {
            console.log(res.data)
            temp_feedback = res.data[0]
            improvement_feedback = res.data[1]
            var random_feedback = temp_feedback[Math.floor(Math.random()*temp_feedback.length)];
            var improvement = improvement_feedback[Math.floor(Math.random()*improvement_feedback.length)]
            console.log(random_feedback)
            if(random_feedback.category == "negative")
            {
              this.setState({
                improvement: true,
              })
            }
            if(testGrade == "A" || testGrade == "B")
            {
              console.log("ff")
              this.setState({
                feedbackColor: 'green',
              })
            }
            else if(testGrade == "C")
            {
              this.setState({
                feedbackColor: '#FFBF00',
              })
            }
            else
            {
              this.setState({
                feedbackColor: 'red',
              })
            }
            this.setState({
                feedbackData: random_feedback,
                improvementFeedbackData: improvement,
                showFeedback: true
            })
            
        })
        
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
    handleEditedFeedbackSubmit(e, requestMethod, test)
    {
      e.preventDefault();
      const feedback = e.target.elements.name.value;
      console.log(feedback);
      this.setState({
        editedFeedbackGiven: true,
        editedFeedback: feedback
      })
    }
    generateFeedback(testName, testGrade, testMark, correct, incorrect, effect){
        console.log("hi yes!!")
        
        this.setState({
            showFeedback: false,
            giveImprovement: false
        })
        axios.get(`http://127.0.0.1:8000/api/processnltk/${testName}/${testGrade}/${testMark}/${correct}/${incorrect}/${effect}`)
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
    generateImprovementfeedback(testName, testGrade, testMark, correct, incorrect, effect){   
        this.setState({
            giveImprovement: true,
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
          <h2 style={{display: 'flex', justifyContent:'center'}} >Create a improvement feedback for{test}here!</h2>
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
    
    onChangeFeedback(e){
      const {name, value} = e.target;
      this.setState({
        editedFeedback: value
      })
    }
    updateFeedback = (e) => {
      this.setState({
        editedFeedback: e
      })
      let secondsToGo = 600;
      const modal = Modal.success({
        title: `You have 10 minutes to edit this feedback otherwise this message will be destroyed` ,
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
          <h2 style={{display: 'flex', justifyContent: 'center'}} >Edit this feedback here!</h2>
          <Form onSubmit={(event) => this.handleEditedFeedbackSubmit(event, e)}>
              <Form.Item label="Feedback">
                <Input name="name" value={this.state.editedFeedback} 
                      onChange={this.onChangeFeedback}placeholder={e} />
              </Form.Item>
              <Form.Item>
                <Button type="primary" htmlType="submit">Publish edited feedback</Button>
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
      let score_one = Math.round(this.state.feedbackData.score * 100)
      let score_two = Math.round(this.state.improvementFeedbackData.score * 100)
      let score1 = this.props.match.params.correct;
      let score2 = this.props.match.params.incorrect;
      let total_Marks = score1+score2;
      var list_topics = new Array();
      var list_marks = new Array();
      list_topics = this.state.topics.split(",")
      list_marks = this.state.marks.split(",")
      const data = [];
      list_topics.map((i,j) =>{
        data.push({title: list_topics[j], mark: list_marks[j]})
      })
      console.log(list_topics)
        return(
            <div className="reviewdiv"> 
                <Row gutter={10} justify="space-around" type="flex">
                  <Col span={6}>
                    <Card className="popupreview" bordered style={{color: 'blue', textAlign: 'center', fontSize: '28px'}} title="Test Information" bordered={false}>
                    <Text strong style={{textAlign: 'center', color: '#096dd9'}}>{this.props.match.params.testid}</Text> <br/>
                    <Row gutter={16}>
                      <Col span={12}>
                        <Statistic title="Test Grade" value={this.props.match.params.testgrade} suffix={" "} />
                      </Col>
                      <Col span={12}>
                        <Statistic title="Test Mark" value={this.props.match.params.testmark} suffix={" %"} />
                      </Col>
                      <Col span={12}>
                        <Statistic title="Correct answers" value={this.props.match.params.correct} suffix={" "} />
                      </Col>
                      <Col span={12}>
                        <Statistic title="Incorrect answers" value={this.props.match.params.incorrect} suffix={" "} />
                      </Col>
                      <Col span={25}>
                        <Statistic  title="Type of test" value={this.state.question_type} suffix={" "} />
                      </Col>
                    </Row>
                    </Card>
                  </Col>
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
                </Row>
                
                <br/>
                {this.state.feedbackColor == "green" 
                
                  ? 
                  <Row>
                  <Col span={8}>
                    <Card className="popupreview" headStyle={{backgroundColor: 'green'}} bordered style={{color: 'blue'}} title="Overall feedback" bordered={false}>
                      <Tooltip title="Click to edit this feedback" onClick={(e) => this.updateFeedback(this.state.feedbackData.review)}>{this.state.editedFeedbackGiven ? <Text strong>Feedback: <Text strong style={{color: '#096dd9'}}>{this.state.editedFeedback}</Text> </Text>: <Text strong >Feedback: {this.state.showFeedback ? <Text strong style={{color: '#096dd9'}}>{this.state.feedbackData.review }</Text>: <Spin indicator={antIcon} />}</Text>}</Tooltip>
                    </Card>
                  </Col>
                  <Col span={10}>
                    <Card className="popupreview" bordered style={{color: 'blue', marginLeft: 150}} 
                          title="Overall feedback information" bordered={false}>
                      <Text strong>Percentage:</Text> {this.state.showFeedback ? 
                      <Text strong style={{color: '#096dd9'}}>{score_one}</Text> 
                        : <Spin indicator={antIcon} />} <br/>
                      <Text strong>Outcome of the generator:</Text> 
                      {this.state.showFeedback ? 
                      <Text strong style={{color: '#096dd9'}}>{this.state.feedbackData.category}</Text>
                        : <Spin indicator={antIcon} />} <br />
                      <Text strong>Outcome of the test:</Text> 
                      {this.state.showFeedback ? 
                        <Text strong style={{color: '#096dd9'}}>{this.state.feedbackData.effectiveness}</Text>
                        : <Spin indicator={antIcon} />} <br />
                      <Text strong type="warning">
                        {this.state.showFeedback ? 
                      <p>Based on this information this student 
                          {this.state.improvement ? 
                      <Text strong style={{color: 'red'}}>Needs improvement!</Text> 
                          : <Text strong style={{color: 'green'}}>Is doing very well!</Text>}</p> 
                          : null} </Text>
                    </Card>
                  </Col>
                </Row>
                :
                this.state.feedbackColor == "red"
                ?
                <Row>
                  <Col span={8}>
                    <Card className="popupreview" headStyle={{backgroundColor: 'red'}} bordered style={{color: 'blue'}} title="Overall feedback" bordered={false}>
                      <Tooltip title="Click to edit this feedback" onClick={(e) => this.updateFeedback(this.state.feedbackData.review)}>{this.state.editedFeedbackGiven ? <Text strong>Feedback: <Text strong style={{color: '#096dd9'}}>{this.state.editedFeedback}</Text> </Text>: <Text strong >Feedback: {this.state.showFeedback ? <Text strong style={{color: '#096dd9'}}>{this.state.feedbackData.review }</Text>: <Spin indicator={antIcon} />}</Text>}</Tooltip>
                    </Card>
                  </Col>
                  <Col span={10}>
                    <Card  className="popupreview" bordered style={{color: 'blue', marginLeft: 150}} title="Overall feedback information" bordered={false}>
                      <Text strong>Percentage:</Text> {this.state.showFeedback ? <Text strong style={{color: '#096dd9'}}>{score_one}</Text> : <Spin indicator={antIcon} />} <br/>
                      <Text strong>Outcome of the generator:</Text> {this.state.showFeedback ? <Text strong style={{color: '#096dd9'}}>{this.state.feedbackData.category}</Text>: <Spin indicator={antIcon} />} <br />
                      <Text strong>Outcome of the test:</Text> {this.state.showFeedback ? <Text strong style={{color: '#096dd9'}}>{this.state.feedbackData.effectiveness}</Text>: <Spin indicator={antIcon} />} <br />
                      <Text strong type="warning">{this.state.showFeedback ? <p>Based on this information this student {this.state.improvement ? <Text strong style={{color: 'red'}}>Needs improvement!</Text> : <Text strong style={{color: 'green'}}>Is doing very well!</Text>}</p> : null} </Text>
                    </Card>
                  </Col>
                </Row>
                :
                this.state.feedbackColor == "#FFBF00"
                ?
                <Row>
                  <Col span={8}>
                    <Card className="popupreview" headStyle={{backgroundColor: '#FFBF00'}} bordered style={{color: 'blue'}} title="Overall feedback" bordered={false}>
                      <Tooltip title="Click to edit this feedback" onClick={(e) => this.updateFeedback(this.state.feedbackData.review)}>{this.state.editedFeedbackGiven ? <Text strong>Feedback: <Text strong style={{color: '#096dd9'}}>{this.state.editedFeedback}</Text> </Text>: <Text strong >Feedback: {this.state.showFeedback ? <Text strong style={{color: '#096dd9'}}>{this.state.feedbackData.review }</Text>: <Spin indicator={antIcon} />}</Text>}</Tooltip>
                    </Card>
                  </Col>
                  <Col span={10}>
                    <Card  className="popupreview" bordered style={{color: 'blue', marginLeft: 150}} title="Overall feedback information" bordered={false}>
                      <Text strong>Percentage:</Text> {this.state.showFeedback ? <Text strong style={{color: '#096dd9'}}>{score_one}</Text> : <Spin indicator={antIcon} />} <br/>
                      <Text strong>Outcome of the generator:</Text> {this.state.showFeedback ? <Text strong style={{color: '#096dd9'}}>{this.state.feedbackData.category}</Text>: <Spin indicator={antIcon} />} <br />
                      <Text strong>Outcome of the test:</Text> {this.state.showFeedback ? <Text strong style={{color: '#096dd9'}}>{this.state.feedbackData.effectiveness}</Text>: <Spin indicator={antIcon} />} <br />
                      <Text strong type="warning">{this.state.showFeedback ? <p>Based on this information this student {this.state.improvement ? <Text strong style={{color: 'red'}}>Needs improvement!</Text> : <Text strong style={{color: 'green'}}>Is doing very well!</Text>}</p> : null} </Text>
                    </Card>
                  </Col>
                </Row>
                :
                <Row>
                  <Col span={8}>
                    <Card className="popupreview" headStyle={{backgroundColor: 'grey'}} bordered style={{color: 'blue'}} title="Overall feedback" bordered={false}>
                      <Tooltip title="Click to edit this feedback" onClick={(e) => this.updateFeedback(this.state.feedbackData.review)}>{this.state.editedFeedbackGiven ? <Text strong>Feedback: <Text strong style={{color: '#096dd9'}}>{this.state.editedFeedback}</Text> </Text>: <Text strong >Feedback: {this.state.showFeedback ? <Text strong style={{color: '#096dd9'}}>{this.state.feedbackData.review }</Text>: <Spin indicator={antIcon} />}</Text>}</Tooltip>
                    </Card>
                  </Col>
                  <Col span={10}>
                    <Card  className="popupreview" bordered style={{color: 'blue', marginLeft: 150}} title="Overall feedback information" bordered={false}>
                      <Text strong>Percentage:</Text> {this.state.showFeedback ? <Text strong style={{color: '#096dd9'}}>{score_one}</Text> : <Spin indicator={antIcon} />} <br/>
                      <Text strong>Outcome of the generator:</Text> {this.state.showFeedback ? <Text strong style={{color: '#096dd9'}}>{this.state.feedbackData.category}</Text>: <Spin indicator={antIcon} />} <br />
                      <Text strong>Outcome of the test:</Text> {this.state.showFeedback ? <Text strong style={{color: '#096dd9'}}>{this.state.feedbackData.effectiveness}</Text>: <Spin indicator={antIcon} />} <br />
                      <Text strong type="warning">{this.state.showFeedback ? <p>Based on this information this student {this.state.improvement ? <Text strong style={{color: 'red'}}>Needs improvement!</Text> : <Text strong style={{color: 'green'}}>Is doing very well!</Text>}</p> : null} </Text>
                    </Card>
                  </Col>
                </Row>
                }
                <br/>
                <Row>
                  <Col span={8}>
                    <Card className="popupreview" headStyle={{backgroundColor: 'red'}} bordered style={{color: 'blue'}} title="Improvement Feedback" bordered={false}>     
                        <Text strong>Feedback: {this.state.personalImprovementGiven 
                        ? <Text strong style={{color: '#096dd9'}}>{this.state.personalImprovement}</Text> : <Text strong style={{color: '#096dd9'}}>{this.state.giveImprovement ?  this.state.improvementFeedbackData.review : null}</Text> }</Text>
                        <Text strong>{this.state.showFeedback ? <Button onClick={(e) => this.generateImprovementfeedback(this.props.match.params.testid, this.props.match.params.testgrade, this.props.match.params.testmark, this.props.match.params.correct, this.props.match.params.incorrect, this.props.match.params.effect)}>Find out feedback for areas of improvement</Button> : <Spin indicator={antIcon} />}</Text><br />
                        <Text strong>{this.state.showFeedback ? <Button onClick={(e) => this.createImprovement(this.props.match.params.testid)}>Create your own improvement feedback</Button> : null}</Text>
                    </Card>
                  </Col>
                  <Col span={10}>
                    {
                    this.state.personalImprovementGiven ? 
                    <Card className="popupreview" bordered style={{color: 'blue',marginLeft: 150}} title="Improvement Information" bordered={false}>
                        <Text strong>Area (s) of improvement: </Text>
                        <Text strong style={{color: '#096dd9'}}>{this.state.AreasOfImprovement}</Text><br />
                        <Text strong>Created by: </Text>
                        <Text strong style={{color: '#096dd9'}}>{this.props.match.params.userid}</Text>
                    </Card>
                    :
                    <Card className="popupreview" bordered style={{color: 'blue',marginLeft: 150}} 
                          title="Improvement Information" bordered={false}>
                          <Text strong>Area (s) of improvement:</Text>
                          {this.state.giveImprovement ? 
                          <Text strong style={{color: '#096dd9'}}>
                            {this.state.AreasOfImprovement}
                          </Text> 
                          : <Spin indicator={antIcon} />} <br/>
                          <Text strong>Improvement:</Text> 
                          {this.state.giveImprovement ? 
                          <Text strong style={{color: '#096dd9'}}>
                            {this.state.improvementFeedbackData.level}
                          </Text> 
                          : <Spin indicator={antIcon} />} <br/>
                          <Text strong>Outcome of the generator:</Text> 
                          {this.state.giveImprovement ? 
                          <Text strong style={{color: '#096dd9'}}>
                            {this.state.improvementFeedbackData.category}
                          </Text> : <Spin indicator={antIcon} />} <br />
                    </Card> 
                    }     
                  </Col>      
                </Row>
                <div className="btnFeedback" style={{textAlign: 'center',  marginLeft: "30%",marginRight: '50%', margin: '23px'}}> 
                    <Button style={{ margin: '5px'}} type="primary" onClick={(e) => this.generateFeedback(this.props.match.params.testid, this.props.match.params.testgrade, this.props.match.params.testmark, this.props.match.params.correct, this.props.match.params.incorrect, this.props.match.params.effect)}>Generate another feedback?</Button>
                    {
                      this.state.personalImprovementGiven ?
                      <Link to={`/generatefeedback/` + this.props.match.params.testid + `/` + this.props.match.params.testmark +`/` + this.props.match.params.testgrade + `/` + this.props.match.params.correct +`/`+ this.props.match.params.incorrect +`/` + this.state.feedbackData.score + `/` + this.state.feedbackData.review + `/` + this.state.personalImprovement + `/` + this.state.AreasOfImprovement + `/` + this.props.match.params.effect + `/` + this.props.match.params.userid}><Button style={{marginLeft: '70%', margin: '5px'}} type="primary">Happy to see the full result?</Button></Link>
                      :
                      <Link to={`/generatefeedback/` + this.props.match.params.testid + `/` + this.props.match.params.testmark +`/` + this.props.match.params.testgrade + `/` + this.props.match.params.correct +`/`+ this.props.match.params.incorrect +`/` + this.state.feedbackData.score + `/` + this.state.feedbackData.review + `/` + this.state.improvementFeedbackData.review + `/` + this.state.AreasOfImprovement + `/` + this.props.match.params.effect + `/` + this.props.match.params.userid}><Button style={{marginLeft: '70%', margin: '5px'}} type="primary">Happy to see the full result?</Button></Link>
                    }
                    <Link to={`/createFeedback/` + this.props.match.params.testid + '/' + this.props.match.params.userid}><Button type="primary" htmlType="submit" style={{alignItems:'center'}}>Not happy with any of these feedbacks?</Button></Link>
                </div>
            </div>
        )
    }
}
export default ReviewFeedback