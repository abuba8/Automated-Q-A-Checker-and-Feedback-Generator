import React from "react"
import {Link} from "react-router-dom";
import axios from "axios";
import { Button, Card, Col, Row,notification  } from "antd";
import '../css/Layout.css';
class FeedbackInformation extends React.Component {
    constructor(props)
    {
        super(props)
        this.state = {
            test: [],
            grades: [],
            generatedFeedback: [],
            answers: [],
            showingAlert: false,
            searchText: '',
            searchedColumn: '',
        }
    }
    componentDidMount(){
        axios.all([
            axios.get('http://127.0.0.1:8000/api/test'),
            axios.get('http://127.0.0.1:8000/api/grades'),
            axios.get('http://127.0.0.1:8000/api/generatedFeedback'),
            axios.get('http://127.0.0.1:8000/api/answers'),
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
        }))
    }
    saveFeedback(event1, event2) {
        console.log(event1 + "" + event2)
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
                    created_by: "Feedback generator",
            })
            this.setState({
              showingAlert: true
            });
            setTimeout(() => {
              this.setState({
                showingAlert: false,
              });
            }, 5000);
            
        }
        else
        {
            alert("Could not save the feedback!")
        }
        console.log(test + " " + grade + " " + feedback + " " + user + " " + percentage)
    }
    
      
    render(){
      const openNotification = () => {
        let total = correct + incorrect;
        notification.open({
          message: correct + ' out of ' + total, 
           onClick: () => {
            console.log('Notification Clicked!');
          },
        });
      };
       
        const {id} = this.props.match.params
        console.log(id)
        let test_id = 0
        let test_grade = 0
        let test_mark = 0
        let effect = ""
        let correct = 0
        let total = 0
        let incorrect = 0
        let sub_questions = 0
            //first get the id of the test based on the test name given
        this.state.test.map(function(item, i){
            if(item.name == id)
            {
                test_id = item.id
                sub_questions = item.num_subquestions
            }
        })
        this.state.grades.map(function(item, i){
            if(item.test == test_id)
            {
                test_grade = item.grade
                test_mark = item.grade_mark
                effect = item.effectiveness
            }
        })
        this.state.answers.map(function(item, i){
          if(item.test == test_id)
          {
            total = item.total_mark_for_question
            correct = item.total_sub_marks
            incorrect = item.total_mark_for_question - item.total_sub_marks
          }
        })
        return(
            <div>
              <div className="site-card-wrapper">
                <Row gutter={10}>
                  <Col span={5}>
                    <Card bordered style={{color: 'blue'}} title="Test Name" bordered={false}>
                      {id}
                    </Card>
                  </Col>
                  <Col span={5}>
                    <Card bordered style={{color: 'blue'}} title="Grade given" bordered={false}>
                      {test_grade}
                    </Card>
                  </Col>
                  <Col span={5}>
                    <Card bordered style={{color: 'blue'}} title="Total sub questions" bordered={false}>
                      {sub_questions}
                    </Card>
                  </Col>
                  <Col span={5}>
                    <Card  bordered style={{color: 'blue'}} title="Percentage Mark" bordered={false}>
                      {test_mark}
                    </Card>
                  </Col>
                  <Col span={5}>
                    <Card  bordered style={{color: 'blue'}} title="Correct answers" bordered={false}>
                      {correct}
                    </Card>
                  </Col>
                  <Col span={5}>
                    <Card bordered style={{color: 'blue'}} title="Incorrect answers" bordered={false}>
                      {incorrect}
                    </Card>
                  </Col> <br />
                  <div className="btn">
                    <Button type="primary" onClick={openNotification}>
                        Open the see overall mark
                    </Button> <br /><br/>
                    <Link to={`/reviewFeedback/` + this.props.match.params.id + `/` + test_mark +`/` + test_grade + `/` + correct +`/`+ incorrect +`/` + effect + `/` +this.props.match.params.userid}><Button>Happy to go and generate a feedback?</Button></Link>
                    <br /><br/>
                    <Link to={`/chooseExistingFeedback/` + this.props.match.params.id + `/` + test_mark +`/` + test_grade + `/` + correct +`/`+ incorrect +`/` + effect + `/` + this.props.match.params.userid}><Button style={{marginLeft: '70%', margin: '5px'}} >Choose from a batch of feedbacks?</Button></Link>
                  </div>
                </Row>
              </div>
              
            </div>
        )
    }
}
export default FeedbackInformation